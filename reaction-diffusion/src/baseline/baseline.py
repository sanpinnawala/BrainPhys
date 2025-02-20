import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
import pytorch_lightning as pl
import wandb
from torchdiffeq import odeint
import numpy as np

from baseline.encoder import Encoder
from baseline.decoder import Decoder
from utils.plot import val_plot_recon, test_plot_recon


class Baseline(pl.LightningModule):
    def __init__(self, hyperparams):
        super(Baseline, self).__init__()
        self.save_hyperparameters()
        self.alpha_range_x = hyperparams['alpha_range_x']
        self.alpha_range_y = hyperparams['alpha_range_y']
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
        self.amsgrad = hyperparams['adam']['amsgrad']
        self.encoder = Encoder(hyperparams)
        self.decoder = Decoder(hyperparams)
        self.log_var = torch.nn.Parameter(torch.zeros(1))
        self.sample = None
        # test
        self.infer_samples = hyperparams['infer_samples']
        self.reconstruct = hyperparams['reconstruct']
        self.artefacts = hyperparams['artefacts_dir']

    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(0.5 * logvar)
        # sample from normal
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    @staticmethod
    def kl_divergence(mean, logvar):
        loss = - 0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return loss

    @staticmethod
    def gaussian_nll_loss(x, x_recon, log_var):
        var = torch.exp(log_var).expand_as(x)
        loss = torch.sum(0.5 * (((x - x_recon) ** 2) / var + torch.log(2 * torch.pi * var)))
        return loss

    def forward(self, x, x_t):
        zX_mean, zX_logvar, zY_mean, zY_logvar, zR_mean, zR_logvar = self.encoder(x, x_t)

        zX = self.reparameterize(zX_mean, zX_logvar)  # [B, 1]
        zY = self.reparameterize(zY_mean, zY_logvar)
        zR = self.reparameterize(zR_mean, zR_logvar)

        init_u = x[:, 0, :, :]
        x_recon = self.decoder(zX, zY, zR, init_u) # [B, T, H, W]

        return x, x_recon, zX_mean, zX_logvar, zY_mean, zY_logvar, zR_mean, zR_logvar, zX, zY, zR

    def loss_function(self, x, x_recon, zX_mean, zX_logvar, zY_mean, zY_logvar, zR_mean, zR_logvar):
        # kl divergence
        kl_zX = self.kl_divergence(zX_mean, zX_logvar)
        kl_zY = self.kl_divergence(zY_mean, zY_logvar)
        kl_zR = self.kl_divergence(zR_mean, zR_logvar)

        kl_loss = kl_zX + kl_zY + kl_zR

        # reconstruction loss
        recon_loss = self.gaussian_nll_loss(x, x_recon, self.log_var)

        total_loss = recon_loss + kl_loss # + penalty

        self.log('recon_loss', recon_loss)
        self.log('kl_loss', kl_loss)
        self.log('log_var', self.log_var)

        return total_loss

    def training_step(self, batch, batch_idx):
        x = batch['heat']
        x_t = batch['time']

        x, x_recon, zX_mean, zX_logvar, zY_mean, zY_logvar, zR_mean, zR_logvar, zX, zY, zR = self(x, x_t)

        loss = self.loss_function(x, x_recon, zX_mean, zX_logvar, zY_mean, zY_logvar, zR_mean, zR_logvar)
        self.log('train_loss', loss)

        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', lr)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['heat']
        x_t = batch['time']

        x, x_recon, zX_mean, zX_logvar, zY_mean, zY_logvar, zR_mean, zR_logvar, zX, zY, zR = self(x, x_t)

        loss = self.loss_function(x, x_recon, zX_mean, zX_logvar, zY_mean, zY_logvar, zR_mean, zR_logvar)
        self.log('val_loss', loss)

        if self.trainer.current_epoch == (self.max_epochs - 1):
            val_plot_recon(x, x_t, x_recon, zX, zY, zR, self.artefacts)

        return loss

    def test_step(self, batch, batch_idx):
        x = batch['heat']
        x_t = batch['time']

        recon_samples = []
        if self.reconstruct:
            for i in range(self.infer_samples):
                _, x_recon, _, _, _, _, _, _, _, _, _ = self(x, x_t)
                recon_samples.append(x_recon.detach().cpu().numpy())

            test_plot_recon(x, x_t, recon_samples, self.artefacts)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=(self.betas[0], self.betas[1]), amsgrad=self.amsgrad)
        scheduler = StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)

        return {'optimizer': optimizer,'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch', 'frequency': 1}}


