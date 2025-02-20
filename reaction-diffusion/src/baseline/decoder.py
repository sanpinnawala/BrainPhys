import torch
from torch import nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, hyperparams):
        super(Decoder, self).__init__()
        self.channel = hyperparams['dconv']['channel']
        self.kernel_size = hyperparams['dconv']['kernel_size']
        self.stride = hyperparams['dconv']['stride']
        self.padding = hyperparams['dconv']['padding']
        self.groups = hyperparams['dconv']['groupnorm']['groups']
        self.slope = hyperparams['dconv']['leakyrelu']['slope']
        self.predefined_t = hyperparams['predefined_t']
        self.spatial_points = hyperparams['spatial_points']

        # to expand latent variables to feature space
        self.fc = nn.Linear(628, self.channel[0] * len(self.predefined_t) * 3 * 3)

        # for upsampling
        self.dconv = nn.Sequential(
            # layer 1
            nn.ConvTranspose2d(self.channel[0], self.channel[1], kernel_size=self.kernel_size, stride=self.stride[0], padding=self.padding),
            nn.GroupNorm(4, self.channel[1]),
            nn.LeakyReLU(negative_slope=self.slope),
            # layer 2
            nn.ConvTranspose2d(self.channel[1], self.channel[2], kernel_size=self.kernel_size, stride=self.stride[1], padding=self.padding),
            nn.GroupNorm(4, self.channel[2]),
            nn.LeakyReLU(negative_slope=self.slope),
            # layer 3
            nn.ConvTranspose2d(self.channel[2], self.channel[3], kernel_size=self.kernel_size, stride=self.stride[2], padding=self.padding),
            nn.Tanh()
        )

        self.initialize_weights()

    def initialize_weights(self):
        # dconv layers initialization
        for layer in self.dconv:
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='leaky_relu')
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.GroupNorm):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

        # fc layers initialization
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, zX, zY, zR, init_u):
        init_u = init_u.view(init_u.shape[0], -1)
        z = torch.cat((zX, zY, zR, init_u), dim=1)  # [B, 3]

        fc_out = self.fc(z) # [B, T * C]
        dconv_in = fc_out.reshape(-1, self.channel[0], 3, 3) # [B * T, C, 3, 3]

        dconv_out = self.dconv(dconv_in)
        out = dconv_out.view(z.shape[0], 1, -1, self.spatial_points, self.spatial_points) # [B, 1, T, H, W]

        out = out.squeeze(1)
        return out
