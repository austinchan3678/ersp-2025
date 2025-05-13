
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.optimize import minimize
from .KANLayer import KANLayer

class KAN(nn.Module):
    def __init__(self, width, seed=0, device='cpu', use_qsp=True, **kwargs):
        super(KAN, self).__init__()

        torch.manual_seed(seed)
        self.depth = len(width) - 1
        self.width = width
        self.device = device
        self.use_qsp = use_qsp

        layers = []
        for i in range(self.depth):
            layer = KANLayer(
                in_dim=width[i],
                out_dim=width[i+1],
                device=device,
                **kwargs
            )
            layers.append(layer)

        self.layers = nn.ModuleList(layers)
        self.to(device)

    def forward(self, x, qsp_params=None):
        for layer in self.layers:
            x, _, _, _ = layer(x, qsp_params)
        return x

    def to(self, device):
        super(KAN, self).to(device)
        self.device = device
        for layer in self.layers:
            layer.to(device)
        return self

    def plot(self, folder="./figures", sample=False):
        if not os.path.exists(folder):
            os.makedirs(folder)

        with torch.no_grad():
            for l, layer in enumerate(self.layers):
                for j in range(layer.out_dim):
                    for i in range(layer.in_dim):
                        preacts = layer.last_preacts[:, j, i].cpu().numpy()
                        postacts = layer.last_postacts[:, j, i].cpu().numpy()
                        idx = preacts.argsort()

                        plt.figure(figsize=(2.5, 2.5))
                        if sample:
                            plt.scatter(preacts[idx], postacts[idx], s=10)
                        else:
                            plt.plot(preacts[idx], postacts[idx], lw=2)
                        plt.xticks([])
                        plt.yticks([])
                        plt.title(f"Layer {l}, Neuron {j} <- {i}")
                        plt.savefig(f"{folder}/sp_{l}_{i}_{j}.png", bbox_inches="tight", dpi=150)
                        plt.close()

    def fit_qsp_params(self, x_tensor, y_tensor, initial_params, maxiter=100):
        def qsp_cost_function(qsp_params_flat):
            qsp_params = np.array(qsp_params_flat)
            with torch.no_grad():
                y_pred = self(x_tensor, qsp_params)
                loss = torch.mean((y_pred - y_tensor) ** 2)
            return loss.item()

        result = minimize(
            qsp_cost_function,
            initial_params,
            method='L-BFGS-B',
            options={'maxiter': maxiter}
        )
        return result
