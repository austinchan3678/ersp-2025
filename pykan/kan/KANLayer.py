import torch
import torch.nn as nn
import numpy as np
from .utils import sparse_mask
from .qsp_activation import QSPActivation

class KANLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num=3, k=3,
                 noise_scale=0.3, scale_base_mu=0.0, scale_base_sigma=1.0,
                 scale_sp=1.0, base_fun=None, grid_eps=0.02, grid_range=[-1, 1],
                 sp_trainable=True, sb_trainable=True, sparse_init=False, device=None):

        super(KANLayer, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.in_dim = in_dim
        self.out_dim = out_dim

        if sparse_init:
            self.mask = torch.nn.Parameter(sparse_mask(in_dim, out_dim), requires_grad=False)
        else:
            self.mask = torch.nn.Parameter(torch.ones(in_dim, out_dim), requires_grad=False)

        self.scale_base = torch.nn.Parameter(
            scale_base_mu * 1 / np.sqrt(in_dim) +
            scale_base_sigma * (torch.rand(in_dim, out_dim) * 2 - 1) * 1 / np.sqrt(in_dim),
            requires_grad=sb_trainable
        )

        self.base_fun = base_fun if base_fun is not None else QSPActivation(device=self.device)

        self.last_preacts = None
        self.last_postacts = None

        self.to(self.device)

    def to(self, device):
        super(KANLayer, self).to(device)
        self.device = device
        return self

    def forward(self, x, qsp_params=None):
        batch = x.shape[0]
        preacts = x[:, None, :].expand(batch, self.out_dim, self.in_dim)

        base = self.base_fun(x, params_override=qsp_params)
        base = base[:, :, None].expand(-1, -1, self.out_dim)

        y = self.scale_base[None, :, :] * base * self.mask[None, :, :]
        postacts = y.permute(0, 2, 1)
        y = torch.sum(y, dim=1)

        self.last_preacts = preacts.detach()
        self.last_postacts = postacts.detach()

        return y, preacts, postacts, None

    def get_subset(self, in_id, out_id):
        spb = KANLayer(len(in_id), len(out_id), base_fun=self.base_fun, device=self.device)
        spb.scale_base.data = self.scale_base[in_id][:, out_id]
        spb.mask.data = self.mask[in_id][:, out_id]
        return spb

    def swap(self, i1, i2, mode='in'):
        with torch.no_grad():
            def swap_(data, i1, i2, mode='in'):
                if mode == 'in':
                    data[i1], data[i2] = data[i2].clone(), data[i1].clone()
                elif mode == 'out':
                    data[:, i1], data[:, i2] = data[:, i2].clone(), data[:, i1].clone()

            swap_(self.scale_base.data, i1, i2, mode=mode)
            swap_(self.mask.data, i1, i2, mode=mode)

