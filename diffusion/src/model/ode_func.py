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
    def laplace_x(y, mask, dx):
        kernel = torch.tensor([[1, -2, 1]], dtype=y.dtype, device=y.device).view(1, 1, 1, -1)
        y = y.unsqueeze(1)  # [B, 1, H, W]
        pad = nn.ReflectionPad2d((1, 1, 0, 0))  # pad only in x-direction
        padded_y = pad(y)
        laplace_x = F.conv2d(padded_y, kernel) / (dx ** 2)
        return laplace_x.squeeze(1)  # [B, H, W]

    @staticmethod
    def laplace_y(y, mask, dy):
        kernel = torch.tensor([[1], [-2], [1]], dtype=y.dtype, device=y.device).view(1, 1, -1, 1)
        y = y.unsqueeze(1)  # [B, 1, H, W]
        pad = nn.ReflectionPad2d((0, 0, 1, 1))  # pad only in y-direction
        padded_y = pad(y)
        laplace_y = F.conv2d(padded_y, kernel) / (dy ** 2)
        return laplace_y.squeeze(1)  # [B, H, W]

    @staticmethod
    def mask_aware_laplacian(y, mask, dx):
        total = torch.zeros_like(y)
        count = torch.zeros_like(y)

        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            y_shift = torch.roll(y, shifts=(di, dj), dims=(1, 2))
            mask_shift = torch.roll(mask, shifts=(di, dj), dims=(1, 2))
            valid = mask * mask_shift  # only keep values where both current and neighbor are valid

            total += y_shift * valid
            count += valid

        lap = (total - count * y) / dx ** 2
        return lap

    @staticmethod
    def interior_mask(mask):
        interior_mask = mask.clone()
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            mask_shift = torch.roll(mask, shifts=(di, dj), dims=(1, 2))
            interior_mask = interior_mask * mask_shift

        return interior_mask

    def initialize_weights(self):
            for layer in self.mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='leaky_relu')
                    nn.init.zeros_(layer.bias)

    def forward(self, y, mask, dx, dy, zX):
        batch_size, h, w = y.shape
        # print('y', y)
        # laplace_x = self.laplace_x(y, mask, dx)
        # laplace_y = self.laplace_y(y, mask, dy)

        # du_dt_x = zX.unsqueeze(1) * laplace_x
        # du_dt_y = zX.unsqueeze(1) * laplace_y

        # du_dt = du_dt_x + du_dt_y
        du_dt = zX.unsqueeze(1) * self.mask_aware_laplacian(y, mask, dx)

        return du_dt
