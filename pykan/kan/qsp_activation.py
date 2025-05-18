# qsp_activation.py

import numpy as np
import torch

# -------------------------------
# Quantum gate definitions
# -------------------------------
def rz(theta):
    theta = float(theta)
    return np.array([[np.exp(-1j * theta / 2), 0],
                     [0, np.exp(1j * theta / 2)]], dtype=complex)

def rx(theta):
    theta = float(theta)
    return np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)],
                     [-1j * np.sin(theta / 2), np.cos(theta / 2)]], dtype=complex)

# -------------------------------
# Build QSP unitary
# -------------------------------
def create_qsp_unitary(params, theta, depth=10):
    U = np.eye(2, dtype=complex)
    for i in range(depth):
        U = rz(params[i]) @ rx(theta) @ U
    U = rz(params[-1]) @ U
    return U

'''# -------------------------------
# Compute transition probability: |⟨1|U|0⟩|²
# -------------------------------
def expectation_value(params, theta, depth=10):
    U = create_qsp_unitary(params, theta, depth)
    ket_0 = np.array([1.0, 0.0], dtype=complex)
    ket_1 = np.array([0.0, 1.0], dtype=complex)
    amplitude = np.vdot(ket_1, U @ ket_0)         # ⟨1|U|0⟩
    probability = np.abs(amplitude) ** 2          # |⟨1|U|0⟩|²
    return probability'''

def expectation_value(params, theta, depth=10):
    U = create_qsp_unitary(params, theta, depth)
    ket_0 = np.array([1.0, 0.0], dtype=complex)
    amplitude = np.vdot(ket_0, U @ ket_0)  # ⟨0|U|0⟩
    return np.real(amplitude)

# -------------------------------
# Torch wrapper for QSP activation
# -------------------------------
class QSPActivation:
    def __init__(self, depth=10, device='cpu'):
        self.depth = depth
        self.device = device
        self.params = torch.nn.Parameter(
            torch.tensor(np.random.uniform(0, 2 * np.pi, 2 * depth + 1), dtype=torch.float32),
            requires_grad=False
        )

    def __call__(self, x: torch.Tensor, params_override=None):
        x_np = x.detach().cpu().numpy()
        out = np.zeros_like(x_np)

        # Use externally supplied parameters if provided
        if params_override is not None:
            params_np = np.array(params_override)
        else:
            params_np = self.params.detach().cpu().numpy()

        for i in range(x_np.shape[0]):
            for j in range(x_np.shape[1]):
                out[i, j] = expectation_value(params_np, x_np[i, j], self.depth)

        return torch.tensor(out, dtype=torch.float32, device=self.device)

'''# qsp_activation.py

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator

def create_qsp_circuit(params, theta, depth=10):
    """Create a single-qubit QSP circuit with given parameters and input angle."""
    circuit = QuantumCircuit(1)
    for i in range(depth):
        circuit.rz(params[i], 0)
        circuit.rx(theta, 0)
    circuit.rz(params[-1], 0)
    return circuit

def expectation_value(params, theta, depth=10):
    """Compute expectation value ⟨0|U|0⟩ for QSP circuit with input theta."""
    circuit = create_qsp_circuit(params, theta, depth)
    state = Statevector.from_instruction(circuit)
    projector = Operator(np.array([[1, 0], [0, 0]]))  # |0⟩⟨0|
    return np.real(state.expectation_value(projector))

class QSPActivation:
    def __init__(self, depth=10, device='cpu'):
        self.depth = depth
        self.device = device
        self.params = torch.nn.Parameter(
            torch.tensor(np.random.uniform(0, 2 * np.pi, 2 * depth + 1), dtype=torch.float32),
            requires_grad=False
        )

    def __call__(self, x: torch.Tensor, params_override=None):
        x_np = x.detach().cpu().numpy()
        out = np.zeros_like(x_np)

        # Use externally supplied parameters if provided
        if params_override is not None:
            params_np = np.array(params_override)
        else:
            params_np = self.params.detach().cpu().numpy()

        for i in range(x_np.shape[0]):
            for j in range(x_np.shape[1]):
                out[i, j] = expectation_value(params_np, x_np[i, j], self.depth)

        return torch.tensor(out, dtype=torch.float32, device=self.device)
'''