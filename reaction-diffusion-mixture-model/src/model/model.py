import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
import pytorch_lightning as pl
import wandb
from torchdiffeq import odeint
import numpy as np

from model.encoder import BaseEncoder, LatentEncoder, CategoricalEncoder
from model.ode_func import ODEFunc1, ODEFunc2, ODEFunc3
from utils.plot import val_plot_recon, test_plot_recon, test_plot_param, test_plot_extrap


class VAE(pl.LightningModule):
    def __init__(self, hyperparams):
        super(VAE, self).__init__()
        self.save_hyperparameters()
        self.hyperparams = hyperparams
        self.alpha_range = hyperparams['alpha_range'] # range for diffusion coefficient
        self.reaction_coeff_range = hyperparams['reaction_coeff_range'] # range for reaction coefficient
        self.total_space = hyperparams['total_space'] # spatial length
        self.spatial_points = hyperparams['spatial_points'] # number of spatial discretisation points
        self.total_time = hyperparams['total_time'] # time length
        self.time_points = hyperparams['time_points'] # number of time discretisation points
        self.predefined_t = hyperparams['predefined_t'] # observed time points
        self.max_epochs = hyperparams['max_epochs']
        self.ode_method = hyperparams['ode']['method'] # integration method
        self.ode_rtol = hyperparams['ode']['rtol'] # relative tolerance
        self.ode_atol = hyperparams['ode']['atol'] # absolute tolerance
        self.lr = hyperparams['adam']['lr'] # learning rate
        self.weight_decay = hyperparams['adam']['weight_decay']
        self.betas = hyperparams['adam']['betas']
        self.amsgrad = hyperparams['adam']['amsgrad']
        self.step_size = hyperparams['steplr']['step_size'] # step size for learning rate decay
        self.gamma = hyperparams['steplr']['gamma'] # learning rate decay factor
        self.categorical_encoder = CategoricalEncoder(hyperparams)
        self.base_encoder = BaseEncoder(hyperparams)
        self.latent_heads = nn.ModuleList([LatentEncoder(hyperparams) for k in range(hyperparams['n_components'])])
        self.ode_funcs = nn.ModuleList([ODEFunc1(hyperparams), ODEFunc2(hyperparams), ODEFunc3(hyperparams)])
        self.log_var = torch.nn.Parameter(torch.zeros(1)) # learnable variance for gaussian negative log likelihood
        # mixture model
        self.n_components = hyperparams['n_components'] # number of mixture components
        # test config
        self.infer_samples = hyperparams['infer_samples']
        self.reconstruct = hyperparams['reconstruct']
        self.param_analysis = hyperparams['param_analysis']
        self.extrapolate = hyperparams['extrapolate']
        self.extrap_t = hyperparams['extrap_t']
        self.extrap_time = hyperparams['extrap_time']
        self.extrap_points = hyperparams['extrap_points']
        self.artefacts = hyperparams['artefacts_dir']
        self.data_dir = hyperparams['data_dir']

    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(0.5 * logvar)
        # sample from normal
        eps = torch.randn_like(std)
        z = mean + eps * std
        # log-normal
        exp_z = torch.exp(z)
        return exp_z

    @staticmethod
    def log_normal_prior(range_min, range_max):
        mean = (range_min + range_max) / 2  # log space mean
        var = ((range_max - range_min) ** 2) / 16  # log space variance

        mean_updated = torch.log(mean ** 2 / torch.sqrt(var + mean ** 2))  # normal space mean
        var_updated = torch.log(1 + (var / mean ** 2))  # normal space variance

        return mean_updated, torch.log(var_updated)

    @staticmethod
    def kl_divergence(mean, logvar, prior_mean, prior_logvar):
        loss = - 0.5 * torch.sum(1 + logvar - prior_logvar - ((mean - prior_mean).pow(2) + logvar.exp()) / torch.exp(prior_logvar), dim=1)
        return loss

    @staticmethod
    def kl_categorical(c, num_models):
        c = c.clamp(min=1e-20)  # avoid log(0)
        loss = torch.sum(c * (torch.log(c) - torch.log(torch.full_like(c, 1.0 / num_models))))
        return loss

    @staticmethod
    def gaussian_nll_loss(x, x_recon, log_var):
        # gaussian negative log likelihood
        var = torch.exp(log_var).expand_as(x)
        loss = torch.sum(0.5 * (((x - x_recon) ** 2) / var + torch.log(2 * torch.pi * var)), dim=tuple(range(1, x.ndim)))
        return loss

    def solve_ode(self, k, zX, zR, init_u, mask, t, total_time, time_points):
        # spatial grid
        x = torch.linspace(0., self.total_space, self.spatial_points, device=init_u.device)  # spatial domain
        dx = x[1] - x[0]

        # fine time grid
        t_fine = torch.linspace(0., total_time, time_points, device=init_u.device)
        t = torch.tensor(t, device=t_fine.device, dtype=t_fine.dtype)
        # time indices for t
        indices = torch.searchsorted(t_fine, t)

        # ode for k component
        ode_func = self.ode_funcs[k]

        def func(t, u):
            # torch.autograd.set_detect_anomaly(True)
            return ode_func(u, mask, dx, zX, zR)

        # solve ode
        y_seq = odeint(func, init_u, t_fine, method=self.ode_method, rtol=self.ode_rtol, atol=self.ode_atol) # [T, B, H, W]
        y_seq = y_seq.permute(1, 0, 2, 3) # [B, T, H, W]

        # [B, T_selected, H, W]
        return y_seq[:, indices, :, :]

    def forward(self, x, x_mask, x_t):
        # initial image for each individual
        init_u = x[:, 0, :, :] # [B, H, W]
        # mask for each individual
        mask = x_mask[:, 0, :, :] # [B, H, W]

        # for outputs from each of the k component
        x_recon_all, zX_mean_all, zX_logvar_all, zR_mean_all, zR_logvar_all, zX_all, zR_all = [], [], [], [], [], [], []

        # component weights
        logits = self.categorical_encoder(x, x_t) # [B, K]
        c = F.softmax(logits, dim=-1)

        # shared encoder
        shared_z = self.base_encoder(x, x_t)

        # for each component in mixture model
        for k in range(self.n_components):
            latent_head = self.latent_heads[k]
            zX_mean, zX_logvar, zR_mean, zR_logvar = latent_head(shared_z)

            # sample from latent distribution
            zX = self.reparameterize(zX_mean, zX_logvar)  # [B, 1]
            zR = self.reparameterize(zR_mean, zR_logvar)

            # solve ode
            x_recon = self.solve_ode(k, zX, zR, init_u, mask, self.predefined_t, self.total_time, self.time_points)  # [B, T, H, W]

            x_recon_all.append(x_recon)
            zX_mean_all.append(zX_mean)
            zX_logvar_all.append(zX_logvar)
            zR_mean_all.append(zR_mean)
            zR_logvar_all.append(zR_logvar)
            zX_all.append(zX)
            zR_all.append(zR)

        # print(c)
        return c, x, x_recon_all, zX_mean_all, zX_logvar_all, zR_mean_all, zR_logvar_all, zX_all, zR_all

    def loss_function(self, outputs):
        zX_prior_mean, zX_prior_logvar = self.log_normal_prior(torch.tensor(self.alpha_range[0]), torch.tensor(self.alpha_range[1]))
        zR_prior_mean, zR_prior_logvar = self.log_normal_prior(torch.tensor(self.reaction_coeff_range[0]), torch.tensor(self.reaction_coeff_range[1]))

        c, x, x_recon_all, zX_mean_all, zX_logvar_all, zR_mean_all, zR_logvar_all, zX_all, zR_all = outputs
        batch_size = x.shape[0]

        # initialise losses
        recon_loss_all = torch.zeros(batch_size, self.n_components, device=self.device)
        kl_zX_all = torch.zeros(batch_size, self.n_components, device=self.device)
        kl_zR_all = torch.zeros(batch_size, self.n_components, device=self.device)

        for k in range(self.n_components):
            # kl divergence
            kl_zX_all[:, k] = self.kl_divergence(zX_mean_all[k], zX_logvar_all[k], zX_prior_mean, zX_prior_logvar)
            kl_zR_all[:, k] = self.kl_divergence(zR_mean_all[k], zR_logvar_all[k], zR_prior_mean, zR_prior_logvar)

            # reconstruction loss
            recon_loss_all[:, k] = self.gaussian_nll_loss(x, x_recon_all[k], self.log_var)

        # expectation over q(c|x)
        recon_loss = (-1) * torch.logsumexp(((-1) * recon_loss_all) + torch.log(c), dim=1).sum()
        kl_zX = torch.sum(c * kl_zX_all, dim=1).sum()
        kl_zR = torch.sum(c * kl_zR_all, dim=1).sum()

        kl_cat = self.kl_categorical(c, self.n_components)

        # total loss
        kl_loss = kl_zX + kl_zR + kl_cat
        total_loss = recon_loss + kl_loss

        self.log('recon_loss', recon_loss)
        self.log('kl_loss', kl_loss)
        self.log('log_var', self.log_var)
        self.log('c_entropy', (-c * torch.log(c + 1e-8)).sum(dim=1).mean() / np.log(self.n_components))

        return total_loss

    def training_step(self, batch, batch_idx):
        x = batch['heat']
        x_mask = batch['mask']
        x_t = batch['time']

        outputs = self(x, x_mask, x_t)

        loss = self.loss_function(outputs)
        self.log('train_loss', loss)

        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', lr)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['heat']
        x_mask = batch['mask']
        x_t = batch['time']

        outputs = self(x, x_mask, x_t)

        loss = self.loss_function(outputs)
        self.log('val_loss', loss)

        if self.trainer.current_epoch == (self.max_epochs - 1):
            c, x, x_recon, zX_mean, zX_logvar, zR_mean, zR_logvar, zX, zR = outputs
            # stack to shape [K, B, T, H, W] and permute to [B, K, T, H, W]
            x_recon_stack = torch.stack(x_recon, dim=0).permute(1, 0, 2, 3, 4)
            # get best component index per sample
            c_idx = torch.argmax(c, dim=1)  # [B]
            # reconstructions from the best component
            x_recon_best = x_recon_stack[torch.arange(c_idx.shape[0]), c_idx]  # shape: [B, T, H, W]
            val_plot_recon(x, x_t, x_recon_best, zX, zR, self.artefacts)

        return loss

    def test_step(self, batch, batch_idx):
        x = batch['heat']
        x_mask = batch['mask']
        x_t = batch['time']
        init_u = x[:, 0, :, :]
        mask = x_mask[:, 0, :, :]

        recon_samples = []

        # reconstruction
        if self.reconstruct:
            for i in range(self.infer_samples):
                c, _, x_recon, _, _, _, _, _, _ = self(x, x_mask, x_t)
                x_recon_stack = torch.stack(x_recon, dim=0).permute(1, 0, 2, 3, 4)
                c_idx = torch.argmax(c, dim=1)  # [B]
                #raise Exception(c_idx)
                x_recon_best = x_recon_stack[torch.arange(c_idx.shape[0]), c_idx]  # shape: [B, T, H, W]
                recon_samples.append(x_recon_best.detach().cpu().numpy())
            test_plot_recon(x, x_t, recon_samples, self.artefacts)

        # parameter analysis
        if self.param_analysis:
            c, _, _, _, _, _, _, zX, zR = self(x, x_mask, x_t)
            zX_stack = torch.stack(zX, dim=0).permute(1, 0, 2)
            zR_stack = torch.stack(zR, dim=0).permute(1, 0, 2)
            c_idx = torch.argmax(c, dim=1)  # [B]
            zX_best = zX_stack[torch.arange(c_idx.shape[0]), c_idx]
            zR_best = zR_stack[torch.arange(c_idx.shape[0]), c_idx]
            test_plot_param(c_idx, zX_best, zR_best, self.artefacts, self.data_dir)

        # interpolation and extrapolation
        #if self.extrapolate:
            #c, _, _, _, _, _, _, zX, zR = self(x, x_mask, x_t)
            #full_t = np.sort(np.concatenate((x_t.detach().cpu().numpy()[0], self.extrap_t)))
            #zX_stack = torch.stack(zX, dim=0).permute(1, 0, 2)
            #zR_stack = torch.stack(zR, dim=0).permute(1, 0, 2)
            #c_idx = torch.argmax(c, dim=1)  # [B]
            #zX_best = zX_stack[torch.arange(c_idx.shape[0]), c_idx]
            #zR_best = zR_stack[torch.arange(c_idx.shape[0]), c_idx]
            #x_extrap = self.solve_ode(c_idx, zX_best, zR_best, init_u, mask, full_t, self.extrap_time, self.extrap_points)
            #test_plot_extrap(full_t, x_extrap, self.artefacts, self.data_dir)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=(self.betas[0], self.betas[1]), amsgrad=self.amsgrad)
        scheduler = StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)

        return {'optimizer': optimizer,'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch', 'frequency': 1}}

