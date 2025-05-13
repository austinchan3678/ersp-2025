import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from .KANLayer import KANLayer

class KAN(nn.Module):
    def __init__(self, width, seed=0, device='cpu', use_qsp=False, **kwargs):
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
                **kwargs  # pass scale_base_mu, scale_base_sigma, sb_trainable, etc.
            )
            layers.append(layer)

        self.layers = nn.ModuleList(layers)
        self.to(device)

    def forward(self, x):
        preacts_all = []
        postacts_all = []
        postsplines_all = []

        for layer in self.layers:
            x, preacts, postacts, postspline = layer(x)
            preacts_all.append(preacts)
            postacts_all.append(postacts)
            postsplines_all.append(postspline)

        return x  # or return all if needed

    def to(self, device):
        super(KAN, self).to(device)
        self.device = device
        for layer in self.layers:
            layer.to(device)
        return self
    
    def plot(self, folder="./figures", sample=False):
        """ 
        Plot the learned functions (activations) for each neuron in the network.
        
        Args:
            folder (str): Directory to save the plots.
            sample (bool): If True, plot dots instead of a line.
        """
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

    def fit(self, dataset, opt="LBFGS", steps=50, lamb=0.0, verbose=True):
        """
        Train the model on the given dataset.

        Args:
            dataset (dict): Must have 'train_input' and 'train_label' keys with torch tensors.
            opt (str): Optimizer ("LBFGS" or "Adam").
            steps (int): Number of optimization steps.
            lamb (float): L2 regularization coefficient.
            verbose (bool): Print loss during training.
        """
        x = dataset['train_input'].to(self.device)
        y_true = dataset['train_label'].to(self.device)

        if opt == "LBFGS":
            optimizer = torch.optim.LBFGS(self.parameters(), max_iter=steps)
            def closure():
                optimizer.zero_grad()
                y_pred = self(x)
                loss = torch.mean((y_pred - y_true) ** 2)
                if lamb > 0:
                    reg = sum(torch.sum(p ** 2) for p in self.parameters())
                    loss += lamb * reg
                loss.backward()
                return loss
            optimizer.step(closure)

        elif opt == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
            for step in range(steps):
                optimizer.zero_grad()
                y_pred = self(x)
                loss = torch.mean((y_pred - y_true) ** 2)
                if lamb > 0:
                    reg = sum(torch.sum(p ** 2) for p in self.parameters())
                    loss += lamb * reg
                loss.backward()
                optimizer.step()
                if verbose and step % 10 == 0:
                    print(f"Step {step}: Loss = {loss.item():.6f}")
        else:
            raise ValueError(f"Unknown optimizer: {opt}")
