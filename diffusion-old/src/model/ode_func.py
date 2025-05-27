import torch
from torch import nn
import torch.nn.functional as F


class ODEFunc(nn.Module):
    def __init__(self, hyperparams):
        super(ODEFunc, self).__init__()
        self.spatial_points = hyperparams['spatial_points']

        self.mlp = nn.Sequential(
            nn.Linear(self.spatial_points * self.spatial_points, self.spatial_points * self.spatial_points),
            nn.Sigmoid()
        )

        self.initialize_weights()

    @staticmethod
    def laplace_x(y, dx):
        kernel = torch.tensor([[1, -2, 1]], dtype=y.dtype, device=y.device).view(1, 1, 1, -1)
        y = y.unsqueeze(1)  # [B, 1, H, W]
        pad = nn.ReflectionPad2d((1, 1, 0, 0))  # pad only in x-direction
        padded_y = pad(y)
        laplace_x = F.conv2d(padded_y, kernel) / (dx ** 2)
        return laplace_x.squeeze(1)  # [B, H, W]

    @staticmethod
    def laplace_y(y, dy):
        kernel = torch.tensor([[1], [-2], [1]], dtype=y.dtype, device=y.device).view(1, 1, -1, 1)
        y = y.unsqueeze(1)  # [B, 1, H, W]
        pad = nn.ReflectionPad2d((0, 0, 1, 1))  # pad only in y-direction
        padded_y = pad(y)
        laplace_y = F.conv2d(padded_y, kernel) / (dy ** 2)
        return laplace_y.squeeze(1)  # [B, H, W]

    def initialize_weights(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='leaky_relu')
                nn.init.zeros_(layer.bias)

    def forward(self, y, dx, dy, zX):
        batch_size, h, w = y.shape
        # print('y', y)
        laplace_x = self.laplace_x(y, dx)
        laplace_y = self.laplace_y(y, dy)

        du_dt_x = zX.unsqueeze(1) * laplace_x
        du_dt_y = zX.unsqueeze(1) * laplace_y
        # anisotropic diffusion-old
        du_dt = du_dt_x + du_dt_y

        return du_dt
