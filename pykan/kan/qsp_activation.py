# qsp_activation.py

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator

def create_qsp_circuit(params, theta, depth=10):
    circuit = QuantumCircuit(1)
    for i in range(depth):
        circuit.rz(params[i], 0)
        circuit.rx(theta, 0)
    circuit.rz(params[-1], 0)
    return circuit

def expectation_value(params, theta, depth=10):
    circuit = create_qsp_circuit(params, theta, depth)
    state = Statevector.from_instruction(circuit)
    projector = Operator(np.array([[1, 0], [0, 0]]))  # |0><0|
    expectation = state.expectation_value(projector)
    return np.real(expectation)

class QSPActivation:
    def __init__(self, depth=10, device='cpu'):
        self.depth = depth
        self.device = device
        self.params = torch.nn.Parameter(
            torch.tensor(np.random.uniform(0, 2 * np.pi, 2 * depth + 1), dtype=torch.float32),
            requires_grad=False
        )

    def __call__(self, x: torch.Tensor):
        x_np = x.detach().cpu().numpy()
        out = np.zeros_like(x_np)
        params_np = self.params.detach().cpu().numpy()
        for i in range(x_np.shape[0]):
            for j in range(x_np.shape[1]):
                out[i, j] = expectation_value(params_np, x_np[i, j], self.depth)
        return torch.tensor(out, dtype=torch.float32, device=self.device)