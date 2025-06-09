'''##### ORIGINAL
import numpy as np
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

    def forward(self, x, qsp_params, alphas, bias=0.0):
        batch_size = x.shape[0]
        preds = []
        for i in range(batch_size):
            x_i = x[i]
            theta = x_i[0].item() if x_i.shape[0] == 1 else x_i[0].item() * x_i[1].item()
            alpha = alphas[i]
            qsp_val = expectation_value(qsp_params, theta, depth=self.qsp_depth)
            preds.append(alpha * qsp_val + bias)
        return torch.tensor(preds, dtype=torch.float32).unsqueeze(1)

    def fit_qsp_with_alphas(self, x_tensor, y_tensor, maxiter=100, cost_fn=None, init_params=None, ftol=1e-9):
        x_np = x_tensor.cpu().numpy()
        y_np = y_tensor.cpu().numpy()
        n = x_np.shape[0]

        if init_params is None:
            init_qsp_params = np.linspace(0, np.pi, self.num_qsp_params)
            init_alphas = np.random.uniform(0.5, 1.5, n)
            init_bias = np.array([0.0])
            init_params = np.concatenate([init_qsp_params, init_alphas, init_bias])

        if cost_fn is None:
            def cost_fn(params):
                qsp_params = params[:self.num_qsp_params]
                alphas = params[self.num_qsp_params:-1]
                bias = params[-1]
                preds = []
                for x, alpha in zip(x_np, alphas):
                    theta = x[0] if len(x) == 1 else x[0] * x[1]
                    preds.append(alpha * expectation_value(qsp_params, theta, depth=self.qsp_depth) + bias)
                return np.mean((y_np.ravel() - np.array(preds)) ** 2)

        result = minimize(cost_fn, init_params, method='L-BFGS-B', options={'maxiter': maxiter, 'ftol': ftol})
        return result'''

'''#################### WEIGHTED COST FN
import numpy as np
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

    def forward(self, x, qsp_params, alphas, bias=0.0):
        batch_size = x.shape[0]
        preds = []
        for i in range(batch_size):
            x_i = x[i]
            theta = x_i[0].item() if x_i.shape[0] == 1 else x_i[0].item() * x_i[1].item()
            alpha = alphas[i]
            qsp_val = expectation_value(qsp_params, theta, depth=self.qsp_depth)
            preds.append(alpha * qsp_val + bias)
        return torch.tensor(preds, dtype=torch.float32).unsqueeze(1)

    def fit_qsp_with_alphas(self, x_tensor, y_tensor, maxiter=100, cost_fn=None, init_params=None, ftol=1e-9):
        x_np = x_tensor.cpu().numpy()
        y_np = y_tensor.cpu().numpy()
        n = x_np.shape[0]

        if init_params is None:
            init_qsp_params = np.linspace(0, np.pi, self.num_qsp_params)
            init_alphas = np.random.uniform(0.5, 1.5, n)
            init_bias = np.array([0.0])
            init_params = np.concatenate([init_qsp_params, init_alphas, init_bias])

        if cost_fn is None:
            def cost_fn(params):
                qsp_params = params[:self.num_qsp_params]
                alphas = params[self.num_qsp_params:-1]
                bias = params[-1]
                preds = []
                for x, alpha in zip(x_np, alphas):
                    theta = x[0] if len(x) == 1 else x[0] * x[1]
                    preds.append(alpha * expectation_value(qsp_params, theta, depth=self.qsp_depth) + bias)
                return np.mean((y_np.ravel() - np.array(preds)) ** 2)

        result = minimize(cost_fn, init_params, method='L-BFGS-B', options={'maxiter': maxiter, 'ftol': ftol})
        return result'''

'''##### LAST WORKING MODEL
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize
from .qsp_activation import expectation_value
from .KANLayer import KANLayer

class KAN(nn.Module):
    def __init__(self, width, qsp_depth=35, device='cpu'):
        super().__init__()
        self.depth = len(width) - 1
        self.width = width
        self.device = torch.device(device)
        self.qsp_depth = qsp_depth
        self.num_qsp_params = 2 * qsp_depth + 1

        self.layers = nn.ModuleList([
            KANLayer(in_dim=width[i], out_dim=width[i+1], device=self.device)
            for i in range(self.depth)
        ])

        self.to(self.device)

    def forward(self, x, qsp_params, alphas, bias=0.0):
        batch_size = x.shape[0]
        preds = []
        for i in range(batch_size):
            theta = x[i][0].item()
            alpha = alphas[i]
            qsp_val = expectation_value(qsp_params, theta, depth=self.qsp_depth)
            preds.append(alpha * qsp_val + bias)
        return torch.tensor(preds, dtype=torch.float32, device=self.device).unsqueeze(1)

    def fit_qsp_with_alphas(self, x_tensor, y_tensor, maxiter=1000, ftol=1e-8,
                             lambda_reg=1e-4, weighted_loss=True, init_params=None):
        x_np = x_tensor.cpu().numpy()
        y_np = y_tensor.cpu().numpy().flatten()
        n = len(x_np)

        if init_params is None:
            init_qsp_params = np.random.uniform(-np.pi, np.pi, self.num_qsp_params)
            init_alphas = np.ones(n)
            init_bias = np.array([0.0])
            init_params = np.concatenate([init_qsp_params, init_alphas, init_bias])

        def cost_fn(params):
            qsp_params = params[:self.num_qsp_params]
            alphas = params[self.num_qsp_params:-1]
            bias = params[-1]
            preds, weights = [], []
            for i in range(n):
                theta = x_np[i][0]
                val = expectation_value(qsp_params, theta, depth=self.qsp_depth)
                preds.append(alphas[i] * val + bias)
                weights.append(1 + 20 * abs(theta)) if weighted_loss else weights.append(1)
            preds = np.array(preds)
            weights = np.array(weights)
            mse = np.mean(weights * (y_np - preds) ** 2)
            reg = lambda_reg * np.mean(np.square(alphas))
            return mse + reg

        result = minimize(cost_fn, init_params, method='L-BFGS-B',
                          options={'maxiter': maxiter, 'ftol': ftol})
        return result'''


import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize
from .qsp_activation import expectation_value
from .KANLayer import KANLayer

class KAN(nn.Module):
    def __init__(self, width, qsp_depth=27, device='cpu'):
        super().__init__()
        self.depth = len(width) - 1
        self.width = width
        self.device = torch.device(device)
        self.qsp_depth = qsp_depth
        self.num_qsp_params = 2 * qsp_depth + 1

        self.layers = nn.ModuleList([
            KANLayer(in_dim=width[i], out_dim=width[i+1], device=self.device)
            for i in range(self.depth)
        ])

        self.to(self.device)

    def forward(self, x, qsp_params, alphas, bias=0.0):
        batch_size = x.shape[0]
        preds = []
        for i in range(batch_size):
            theta = x[i][0].item()
            alpha = alphas[i]
            qsp_val = expectation_value(qsp_params, theta, depth=self.qsp_depth)
            preds.append(alpha * qsp_val + bias)
        return torch.tensor(preds, dtype=torch.float32, device=self.device).unsqueeze(1)

    def fit_qsp_with_alphas(self, x_tensor, y_tensor, maxiter=1000, ftol=1e-8,
                             lambda_reg=1e-4, weighted_loss=True, init_params=None):
        x_np = x_tensor.cpu().numpy()
        y_np = y_tensor.cpu().numpy().flatten()
        n = len(x_np)

        if init_params is None:
            init_qsp_params = np.random.uniform(-np.pi, np.pi, self.num_qsp_params)
            init_alphas = np.ones(n)
            init_bias = np.array([0.0])
            init_params = np.concatenate([init_qsp_params, init_alphas, init_bias])

        def cost_fn(params):
            qsp_params = params[:self.num_qsp_params]
            alphas = params[self.num_qsp_params:-1]
            bias = params[-1]
            preds, weights = [], []
            for i in range(n):
                theta = x_np[i][0]
                val = expectation_value(qsp_params, theta, depth=self.qsp_depth)
                preds.append(alphas[i] * val + bias)
                weights.append(1 + 10 * np.exp(-50 * theta**2)) if weighted_loss else weights.append(1)
            preds = np.array(preds)
            weights = np.array(weights)
            mse = np.mean(weights * (y_np - preds) ** 2)
            reg = lambda_reg * np.mean(np.square(alphas))
            return mse + reg

        result = minimize(cost_fn, init_params, method='L-BFGS-B',
                          options={'maxiter': maxiter, 'ftol': ftol})
        return result

    def prune_internal_nodes(self, threshold=0.01):
        for i, layer in enumerate(self.layers):
            layer.prune_weights(threshold)




'''# WITH PRUNING FOR REGIONS 
import numpy as np
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

        self.grid = None
        self.num_regions = 0

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

    def set_grid(self, x_full, num_regions=3):
        splits = np.linspace(np.min(x_full), np.max(x_full), num_regions + 1)
        self.grid = [(splits[i], splits[i + 1]) for i in range(num_regions)]
        self.num_regions = num_regions

    def forward(self, x, all_qsp_params, alphas, bias=0.0):
        preds = []
        for i in range(x.shape[0]):
            xi = x[i][0].item()
            alpha = alphas[i]
            region_idx = next(j for j, (a, b) in enumerate(self.grid) if a <= xi < b)
            qsp_params = all_qsp_params[region_idx]
            qsp_val = expectation_value(qsp_params, xi, depth=self.qsp_depth)
            preds.append(alpha * qsp_val + bias)
        return torch.tensor(preds, dtype=torch.float32).unsqueeze(1)

    def fit_grid_qsp_with_alphas(self, x_tensor, y_tensor, maxiter=1000, lambda_reg=1e-3, ftol=1e-9):
        x_np = x_tensor.cpu().numpy()
        y_np = y_tensor.cpu().numpy()
        n = x_np.shape[0]
        num_qsp = self.num_qsp_params
        num_regions = self.num_regions

        init_qsp_params = np.concatenate([np.linspace(0, np.pi, num_qsp) for _ in range(num_regions)])
        init_alphas = np.random.uniform(0.5, 1.5, n)
        init_params = np.concatenate([init_qsp_params, init_alphas])

        def cost_fn(params):
            qsp_param_sets = [params[i*num_qsp:(i+1)*num_qsp] for i in range(num_regions)]
            alphas = params[num_regions*num_qsp:]
            preds, weights = [], []
            for i in range(len(x_np)):
                xi = x_np[i][0]
                yi = y_np[i][0]
                for j, (a, b) in enumerate(self.grid):
                    if (a <= xi < b) or (j == len(self.grid) - 1 and a <= xi <= b):
                        region_idx = j
                        break
                else:
                    raise ValueError(f"Value {xi} does not fall into any grid region: {self.grid}")

                val = expectation_value(qsp_param_sets[region_idx], xi, depth=self.qsp_depth)
                preds.append(alphas[i] * val)
                weights.append(1 + 10 * (abs(xi) ** 2))
            preds = np.array(preds)
            weights = np.array(weights)
            mse = np.mean(weights * (y_np.flatten() - preds) ** 2)
            penalty = lambda_reg * np.mean(np.square(alphas))
            return mse + penalty

        result = minimize(cost_fn, init_params, method='L-BFGS-B', options={'maxiter': maxiter, 'ftol': ftol})
        return result'''

'''####### GRID LOGIC: BAD ######################################
import numpy as np
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
        self.grid = []
        self.to(device)

    def forward(self, x, qsp_params_list, alphas, bias=0.0):
        preds = []
        for i in range(x.shape[0]):
            theta = x[i, 0].item()
            region_idx = self.get_region_index(theta)
            qsp_params = qsp_params_list[region_idx]
            alpha = alphas[i]
            qsp_val = expectation_value(qsp_params, theta, depth=self.qsp_depth)
            preds.append(alpha * qsp_val + bias)
        return torch.tensor(preds, dtype=torch.float32).unsqueeze(1)

    def set_grid(self, x, num_regions=3):
        x = np.sort(x.flatten())
        boundaries = np.linspace(x[0], x[-1], num_regions + 1)
        self.grid = [(boundaries[i], boundaries[i + 1]) for i in range(num_regions)]

    def get_region_index(self, xi):
        epsilon = 1e-6  # increased tolerance
    
        # Clamp xi to be within the grid bounds
        xi = max(min(xi, self.grid[-1][1] - epsilon), self.grid[0][0] + epsilon)
    
        for j, (a, b) in enumerate(self.grid):
            if a - epsilon <= xi < b or (j == len(self.grid) - 1 and a - epsilon <= xi <= b + epsilon):
                return j
    
        raise ValueError(f"Value {xi} not in any grid region: {self.grid}")



    def fit_grid_qsp_with_alphas(self, x_tensor, y_tensor, maxiter=1000, lambda_reg=1e-3, ftol=1e-9):
        x_np = x_tensor.cpu().numpy()
        y_np = y_tensor.cpu().numpy()
        n = x_np.shape[0]
        num_regions = len(self.grid)
        num_qsp = self.num_qsp_params

        qsp_params_flat = np.tile(np.linspace(0, np.pi, num_qsp), num_regions)
        alphas = np.random.uniform(0.5, 1.5, n)
        init_params = np.concatenate([qsp_params_flat, alphas])

        def cost_fn(params):
            qsp_param_sets = [params[i*num_qsp:(i+1)*num_qsp] for i in range(num_regions)]
            alphas = params[num_regions*num_qsp:]
            preds = []
            for i in range(n):
                xi = x_np[i][0]
                yi = y_np[i][0]
                region_idx = self.get_region_index(xi)
                val = expectation_value(qsp_param_sets[region_idx], xi, depth=self.qsp_depth)
                preds.append(alphas[i] * val)
            preds = np.array(preds)
            mse = np.mean((y_np.flatten() - preds)**2)
            penalty = lambda_reg * np.mean(np.square(alphas))
            return mse + penalty

        result = minimize(cost_fn, init_params, method='L-BFGS-B', options={'maxiter': maxiter, 'ftol': ftol})
        return result'''
