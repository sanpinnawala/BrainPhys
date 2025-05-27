import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
import pytorch_lightning as pl
import wandb
from torchdiffeq import odeint
import numpy as np

from model.encoder import LatentEncoder, CategoricalEncoder
from model.ode_func import ODEFunc1, ODEFunc2
from utils.plot import val_plot_recon, test_plot_recon, test_plot_param, test_plot_extrap


class VAE(pl.LightningModule):
    def __init__(self, hyperparams):
        super(VAE, self).__init__()
        self.save_hyperparameters()
        self.hyperparams = hyperparams
        self.alpha_range = hyperparams['alpha_range']
        self.reaction_coeff_range = hyperparams['reaction_coeff_range']
        self.total_space = hyperparams['total_space']
        self.spatial_points = hyperparams['spatial_points']
        self.total_time = hyperparams['total_time']
        self.time_points = hyperparams['time_points']
        self.predefined_t = hyperparams['predefined_t']
        self.max_epochs = hyperparams['max_epochs']
        self.ode_method = hyperparams['ode']['method']
        self.ode_rtol = hyperparams['ode']['rtol']
        self.ode_atol = hyperparams['ode']['atol']
        self.lr = hyperparams['adam']['lr']
        self.weight_decay = hyperparams['adam']['weight_decay']
        self.betas = hyperparams['adam']['betas']
        self.amsgrad = hyperparams['adam']['amsgrad']
        self.step_size = hyperparams['steplr']['step_size']
        self.gamma = hyperparams['steplr']['gamma']
        self.monotonic_regularisation = hyperparams['monotonic_regularisation']
        self.reg_lambda = hyperparams['reg_lambda']
        self.categorical_encoder = CategoricalEncoder(hyperparams)
        self.latent_encoder = LatentEncoder(hyperparams)
        # self.latent_encoders = nn.ModuleList([LatentEncoder(hyperparams) for k in range(hyperparams['n_components'])])
        # self.ode_funcs = nn.ModuleList([ODEFunc(hyperparams) for k in range(hyperparams['n_components'])])
        self.ode_funcs = nn.ModuleList([ODEFunc1(hyperparams), ODEFunc2(hyperparams)])
        self.log_var = torch.nn.Parameter(torch.zeros(1))
        self.n_components = hyperparams['n_components']
        self.tau = hyperparams['tau']
        # test
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
    def gumbel_softmax(logits, tau):
        gumbel = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
        c = F.softmax((logits + gumbel)/tau, dim=-1)
        return c

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

        mean_updated = np.log(mean ** 2 / np.sqrt(var + mean ** 2))  # normal space mean
        var_updated = np.log(1 + (var / mean ** 2))  # normal space variance

        return torch.tensor(mean_updated), torch.log(torch.tensor(var_updated))

    @staticmethod
    def kl_divergence(mean, logvar, prior_mean, prior_logvar):
        loss = - 0.5 * torch.sum(1 + logvar - prior_logvar - ((mean - prior_mean).pow(2) + logvar.exp()) / torch.exp(prior_logvar))
        return loss

    @staticmethod
    def kl_categorical(c, num_models):
        c = c.clamp(min=1e-20)  # avoid log(0)
        loss = torch.sum(c * (torch.log(c) - torch.log(torch.full_like(c, 1.0 / num_models))))
        return loss

    @staticmethod
    def gaussian_nll_loss(x, x_recon, log_var):
        var = torch.exp(log_var).expand_as(x)
        loss = torch.sum(0.5 * (((x - x_recon) ** 2) / var + torch.log(2 * torch.pi * var)))
        return loss

    def solve_ode(self, k, zX, zR, init_u, mask, t, total_time, time_points):
        x = torch.linspace(0., self.total_space, self.spatial_points, device=init_u.device)  # spatial domain

        dx = x[1] - x[0]

        t_fine = torch.linspace(0., total_time, time_points, device=init_u.device)
        t = torch.tensor(t, device=t_fine.device, dtype=t_fine.dtype)
        indices = torch.searchsorted(t_fine, t)

        ode_func = self.ode_funcs[k]

        def func(t, u):
            # torch.autograd.set_detect_anomaly(True)
            return ode_func(u, mask, dx, zX, zR)

        # solve ode
        y_seq = odeint(func, init_u, t_fine, method=self.ode_method, rtol=self.ode_rtol, atol=self.ode_atol)
        y_seq = y_seq.permute(1, 0, 2, 3) # [B, T, H, W]

        return y_seq[:, indices, :, :]

    def forward(self, x, x_mask, x_t):
        b, t, h, w = x.shape
        init_u = x[:, 0, :, :]
        mask = x_mask[:, 0, :, :]

        logits = self.categorical_encoder(x, x_t)
        c = self.gumbel_softmax(logits, self.tau) # [B, 4]

        zX_mean, zX_logvar, zR_mean, zR_logvar = self.latent_encoder(c, x, x_t)

        zX = self.reparameterize(zX_mean, zX_logvar)  # [B, 1]
        zR = self.reparameterize(zR_mean, zR_logvar)

        x_recons = []

        for k in range(self.n_components):
            x_recon = self.solve_ode(k, zX, zR, init_u, mask, self.predefined_t, self.total_time, self.time_points)  # [B, T, H, W]
            x_recons.append(x_recon)
        print(c)
        return c, x, x_recons, zX_mean, zX_logvar, zR_mean, zR_logvar, zX, zR

    def loss_function(self, outputs):
        c, x, x_recons, zX_mean, zX_logvar, zR_mean, zR_logvar, zX, zR = outputs
        b = x.shape[0]

        recon_losses = torch.zeros(b, self.n_components, device=self.device)
        kl_zXs = torch.zeros(b, self.n_components, device=self.device)
        kl_zRs = torch.zeros(b, self.n_components, device=self.device)

        zX_prior_mean, zX_prior_logvar = self.log_normal_prior(self.alpha_range[0], self.alpha_range[1])
        zR_prior_mean, zR_prior_logvar = self.log_normal_prior(self.reaction_coeff_range[0], self.reaction_coeff_range[1])

        for k in range(self.n_components):
            # kl divergence
            kl_zXs[:, k] = self.kl_divergence(zX_mean, zX_logvar, zX_prior_mean, zX_prior_logvar)
            kl_zRs[:, k] = self.kl_divergence(zR_mean, zR_logvar, zR_prior_mean, zR_prior_logvar)

            # reconstruction loss
            recon_losses[:, k] = self.gaussian_nll_loss(x, x_recons[k], self.log_var)

        # expectation over q(c|x)
        recon_loss = torch.sum(c * recon_losses, dim=1).sum()
        kl_zX = torch.sum(c * kl_zXs, dim=1).sum()
        kl_zR = torch.sum(c * kl_zRs, dim=1).sum()

        kl_cat = self.kl_categorical(c, self.n_components)

        kl_loss = kl_zX + kl_zR + kl_cat
        total_loss = recon_loss + kl_loss

        self.log('recon_loss', recon_loss)
        self.log('kl_loss', kl_loss)
        self.log('log_var', self.log_var)

        return total_loss#.sum()

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
            c, x, x_recons, zX_means, zX_logvars, zR_means, zR_logvars, zXs, zRs = outputs
            val_plot_recon(x, x_t, x_recons[0], zXs[0], zRs[0], self.artefacts)

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
                _, x_recon, _, _, _, _, _, _ = self(x, x_mask, x_t)
                recon_samples.append(x_recon.detach().cpu().numpy())
            test_plot_recon(x, x_t, recon_samples, self.artefacts)

        # parameter analysis
        if self.param_analysis:
            _, _, _, _, _, _, zX, zR = self(x, x_mask, x_t)
            test_plot_param(zX, zR, self.artefacts, self.data_dir)

        # interpolation and extrapolation
        if self.extrapolate:
            _, _, _, _, _, _, zX, zR = self(x, x_mask, x_t)
            full_t = np.sort(np.concatenate((x_t.detach().cpu().numpy()[0], self.extrap_t)))
            x_extrap = self.solve_ode(zX, zR, init_u, mask, full_t, self.extrap_time, self.extrap_points)
            test_plot_extrap(full_t, x_extrap, self.artefacts, self.data_dir)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=(self.betas[0], self.betas[1]), amsgrad=self.amsgrad)
        scheduler = StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)

        return {'optimizer': optimizer,'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch', 'frequency': 1}}

