import torch
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, hyperparams):
        super(Encoder, self).__init__()
        self.channel = hyperparams['conv']['channel']
        self.kernel_size = hyperparams['conv']['kernel_size']
        self.stride = hyperparams['conv']['stride']
        self.padding = hyperparams['conv']['padding']
        self.groups = hyperparams['conv']['groupnorm']['groups']
        self.slope = hyperparams['conv']['leakyrelu']['slope']
        self.pool_kernel_size = hyperparams['conv']['maxpool']['kernel_size']
        self.pool_stride = hyperparams['conv']['maxpool']['stride']
        self.hidden_size = hyperparams['lstm']['hidden_size']
        self.alpha_range_x = hyperparams['alpha_range_x']
        self.alpha_range_y = hyperparams['alpha_range_y']
        self.reaction_coeff_range = hyperparams['reaction_coeff_range']

        # feature extraction
        self.conv = nn.Sequential(
            # layer 1
            nn.Conv2d(self.channel[0], self.channel[1], kernel_size=self.kernel_size, stride=self.stride[0], padding=self.padding),
            nn.GroupNorm(self.groups, self.channel[1]),
            nn.LeakyReLU(negative_slope=0.02),
            nn.MaxPool2d(kernel_size=self.pool_kernel_size, stride=self.pool_stride),
            # layer 2
            nn.Conv2d(self.channel[1], self.channel[2], kernel_size=self.kernel_size, stride=self.stride[1], padding=self.padding),
            nn.GroupNorm(self.groups, self.channel[2]),
            nn.LeakyReLU(negative_slope=0.02),
            nn.MaxPool2d(kernel_size=self.pool_kernel_size, stride=self.pool_stride),
            # layer 3
            nn.Conv2d(self.channel[2], self.channel[3], kernel_size=self.kernel_size, stride=self.stride[2], padding=self.padding),
            nn.GroupNorm(self.groups, self.channel[3]),
            nn.LeakyReLU(negative_slope=0.02),
            nn.AdaptiveAvgPool2d(1),
        )

        # for temporal dependencies
        self.lstm = nn.LSTM(input_size=self.channel[3] * 1 * 1 + 1, hidden_size=self.hidden_size, num_layers=2, batch_first=True)

        # fully connected layers for latent variables
        self.fc_zX_mean = nn.Linear(self.hidden_size, 1)  # for zX mean
        self.fc_zX_logvar = nn.Linear(self.hidden_size, 1)  # for zX logvar

        self.fc_zY_mean = nn.Linear(self.hidden_size, 1)  # for zY mean
        self.fc_zY_logvar = nn.Linear(self.hidden_size, 1)  # for zY logvar

        self.fc_zR_mean = nn.Linear(self.hidden_size, 1)  # for zR mean
        self.fc_zR_logvar = nn.Linear(self.hidden_size, 1)  # for zR logvar

        self.initialize_weights()

    def initialize_weights(self):
        # conv layers initialization
        for layer in self.conv:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='leaky_relu')
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

        # lstm initialization
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        # fc layers initialization
        for layer in [self.fc_zX_mean, self.fc_zX_logvar, self.fc_zY_mean, self.fc_zY_logvar, self.fc_zR_mean, self.fc_zR_logvar]:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

        # set prior means for the latent variables
        zX_prior_mean = (self.alpha_range_x[0] + self.alpha_range_x[1]) / 2
        nn.init.constant_(self.fc_zX_mean.bias, zX_prior_mean)  # set bias for zX mean

        zY_prior_mean = (self.alpha_range_y[0] + self.alpha_range_y[1]) / 2
        nn.init.constant_(self.fc_zY_mean.bias, zY_prior_mean)  # set bias for zY mean

        zR_prior_mean = (self.reaction_coeff_range[0] + self.reaction_coeff_range[1]) / 2
        nn.init.constant_(self.fc_zR_mean.bias, zR_prior_mean)  # set bias for zR mean

    def forward(self, x, x_t):
        x = x.unsqueeze(1)  # [B, 1, T, H, W]

        batch_size, _, t, h, w = x.shape # [B, 1, T, H, W]
        x = x.view(batch_size * t, 1, h, w) # [B * T, 1, H, W]

        conv_out = self.conv(x) # [B * T, C, H, W]
        # raise Exception(conv_out.shape)

        conv_out = conv_out.view(batch_size * t, -1)  # [B * T, C * H * W]
        conv_out = conv_out.view(batch_size, t, -1) # [B, T, C * H * W]

        x_t = x_t.unsqueeze(-1) # [B, T, 1]
        lstm_in = torch.cat([conv_out, x_t], dim=-1)

        _, (h_n, _) = self.lstm(lstm_in) # [B, T, hidden_dim]
        z = h_n[-1]
        # raise Exception(z.shape)

        zX_mean = self.fc_zX_mean(z)
        zX_logvar = self.fc_zX_logvar(z)

        zY_mean = self.fc_zY_mean(z)
        zY_logvar = self.fc_zY_logvar(z)

        zR_mean = self.fc_zR_mean(z)
        zR_logvar = self.fc_zR_logvar(z)

        return zX_mean, zX_logvar, zY_mean, zY_logvar, zR_mean, zR_logvar
