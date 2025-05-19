'''import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import minimize
from .KANLayer import KANLayer
from .qsp_activation import expectation_value

class KAN(nn.Module):
    def __init__(self, width, qsp_depth=10, device='cpu'):
        super().__init__()
        self.depth = len(width) - 1
        self.width = width
        self.device = device
        self.qsp_depth = qsp_depth
        self.num_qsp_params = 2 * qsp_depth + 1

        self.layers = nn.ModuleList([
            KANLayer(width[i], width[i+1], device=device)
            for i in range(self.depth)
        ])
        self.to(device)

    def forward(self, x, qsp_params=None):
        for layer in self.layers:
            x, _, _, _ = layer(x, qsp_params=qsp_params)
        return x

    def fit_qsp_with_alphas(self, x_tensor, y_tensor, maxiter=100):
        x_np = x_tensor.cpu().numpy()
        y_np = y_tensor.cpu().numpy().reshape(-1)
        n = x_np.shape[0]

        init_qsp_params = np.random.uniform(0, 2 * np.pi, self.num_qsp_params)
        init_alphas = np.ones(n)  # start with neutral scaling
        init_params = np.concatenate([init_qsp_params, init_alphas])

        def cost_fn(params):
            qsp_params = params[:self.num_qsp_params]
            alphas = params[self.num_qsp_params:]

            x_torch = torch.tensor(x_np, dtype=torch.float32)
            with torch.no_grad():
                raw_preds = self.forward(x_torch, qsp_params=qsp_params).squeeze().cpu().numpy()
                scaled_preds = alphas * raw_preds
                error = np.mean((y_np - scaled_preds) ** 2)
            return error

        result = minimize(cost_fn, init_params, method='L-BFGS-B', options={'maxiter': maxiter})
        return result'''



# BESSEL KERNEL APPROXIMATION NETWORK (KAN)
import numpy as np
import os
import torch
import torch.nn as nn
from scipy.optimize import minimize
from .qsp_activation import expectation_value
from .KANLayer import KANLayer

class KAN(nn.Module):
    def __init__(self, width, qsp_depth=10, device='cpu'):
        super().__init__()
        self.depth = len(width) - 1
        self.width = width
        self.device = device
        self.qsp_depth = qsp_depth
        self.num_qsp_params = 2 * qsp_depth + 1

        layers = []
        for i in range(self.depth):
            layer = KANLayer(
                in_dim=width[i],
                out_dim=width[i+1],
                device=device
            )
            layers.append(layer)

        self.layers = nn.ModuleList(layers)
        self.to(device)


    def forward(self, x, qsp_params, alphas):
        batch_size = x.shape[0]
        preds = []
        for i in range(batch_size):
            x_i = x[i]
            # Handle 1D input (e.g., [x]) or 2D (e.g., [x, y])
            theta = x_i[0].item() if x_i.shape[0] == 1 else x_i[0].item() * x_i[1].item()
            alpha = alphas[i]
            qsp_val = expectation_value(qsp_params, theta, depth=self.qsp_depth)
            preds.append(alpha * qsp_val)
        return torch.tensor(preds, dtype=torch.float32).unsqueeze(1)


    def fit_qsp_with_alphas(self, x_tensor, y_tensor, maxiter=100, cost_fn=None, init_params=None):
        x_np = x_tensor.cpu().numpy()
        y_np = y_tensor.cpu().numpy()
        n = x_np.shape[0]

        # Use custom initialization if provided
        if init_params is None:
            init_qsp_params = np.linspace(0, np.pi, self.num_qsp_params)
            init_alphas = np.random.uniform(0.5, 1.5, n)
            init_params = np.concatenate([init_qsp_params, init_alphas])

        # Default cost function if none provided
        if cost_fn is None:
            def cost_fn(params):
                qsp_params = params[:self.num_qsp_params]
                alphas = params[self.num_qsp_params:]
                preds = []
                for x, alpha in zip(x_np, alphas):
                    if len(x) == 1:
                        theta = x[0]
                    elif len(x) == 2:
                        theta = x[0] * x[1]
                    else:
                        raise ValueError(f"Unexpected input dimension: {x}")
                    preds.append(alpha * expectation_value(qsp_params, theta, depth=self.qsp_depth))
                return np.mean((y_np.ravel() - np.array(preds)) ** 2)

        result = minimize(cost_fn, init_params, method='L-BFGS-B', options={'maxiter': maxiter})
        return result





