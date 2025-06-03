import torch
from torch import nn
import torch.nn.functional as F


class BaseODEFunc(nn.Module):
    def __init__(self, hyperparams):
        super(BaseODEFunc, self).__init__()
        self.spatial_points = hyperparams['spatial_points']

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

class ODEFunc1(BaseODEFunc):
    def forward(self, y, mask, dx, zX, zR):
        du_dt = zX.unsqueeze(1) * self.mask_aware_laplacian(y, mask, dx)
        du_dt += zR.unsqueeze(1) * y * (1 - y)

        return du_dt

class ODEFunc2(BaseODEFunc):
    def forward(self, y, mask, dx, zX, zR):
        du_dt = zX.unsqueeze(1) * self.mask_aware_laplacian(y, mask, dx)
        du_dt += zR.unsqueeze(1) * y * (1 - y).pow(2)

        return du_dt

class ODEFunc3(BaseODEFunc):
    def forward(self, y, mask, dx, zX, zR):
        du_dt = zX.unsqueeze(1) * self.mask_aware_laplacian(y, mask, dx)
        du_dt += zR.unsqueeze(1) * y.pow(2) * (1 - y)

        return du_dt

