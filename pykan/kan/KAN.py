
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize
from .qsp_activation import expectation_value

class KAN(nn.Module):
    def __init__(self, qsp_depth=10):
        super().__init__()
        self.qsp_depth = qsp_depth
        self.num_qsp_params = 2 * qsp_depth + 1

    def forward(self, x, qsp_params, alphas):
        batch_size = x.shape[0]
        preds = []
        for i in range(batch_size):
            theta = x[i].item()
            alpha = alphas[i]
            qsp_val = expectation_value(qsp_params, theta, depth=self.qsp_depth)
            preds.append(alpha * qsp_val)
        return torch.tensor(preds, dtype=torch.float32).unsqueeze(1)

    def fit_qsp_with_alphas(self, x_tensor, y_tensor, maxiter=100):
        x_np = x_tensor.cpu().numpy().reshape(-1)  # Ensures 1D array
        y_np = y_tensor.cpu().numpy().reshape(-1)
        n = x_np.shape[0]


        init_qsp_params = np.random.uniform(0, 2 * np.pi, self.num_qsp_params)
        init_alphas = np.random.uniform(0.5, 1.5, n)
        init_params = np.concatenate([init_qsp_params, init_alphas])

        def cost_fn(params):
            qsp_params = params[:self.num_qsp_params]
            alphas = params[self.num_qsp_params:]
            preds = [alpha * expectation_value(qsp_params, x) for x, alpha in zip(x_np, alphas)]
            error = np.mean((y_np - np.array(preds)) ** 2)
            return error

        result = minimize(cost_fn, init_params, method='L-BFGS-B', options={'maxiter': maxiter})
        return result
