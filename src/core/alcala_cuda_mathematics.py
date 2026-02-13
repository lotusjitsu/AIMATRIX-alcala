"""
ALCALA CUDA-Accelerated Mathematics Module
Advanced mathematical functions for neural network using CUDA/cuDNN

Integrates with:
- NVIDIA cuDNN Frontend (C++ backend)
- CUDA acceleration for matrix operations
- Advanced mathematical functions for quantitative analysis

This module provides:
1. GPU-accelerated linear algebra
2. Advanced statistical functions
3. Quantum mathematics utilities
4. Trading mathematical indicators
5. Neural network mathematical primitives
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from pathlib import Path
import subprocess
import json


class ALCALACUDAMathematics:
    """
    CUDA-accelerated mathematics for ALCALA neural network

    Provides advanced mathematical functions optimized for GPU
    """

    def __init__(self):
        self.cuda_available = self._check_cuda()
        self.cudnn_frontend_path = Path(__file__).parent.parent.parent / "cudnn-frontend"

        print(f"[ALCALA CUDA Math] Initialization...")
        print(f"[ALCALA CUDA Math] CUDA Available: {self.cuda_available}")
        print(f"[ALCALA CUDA Math] cuDNN Frontend Path: {self.cudnn_frontend_path.exists()}")

        # Mathematical constants
        self.PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
        self.E = np.e
        self.PI = np.pi

    def _check_cuda(self) -> bool:
        """Check if CUDA is available"""
        try:
            # Try to import torch or cupy if available
            try:
                import torch
                return torch.cuda.is_available()
            except ImportError:
                pass

            try:
                import cupy
                return True
            except ImportError:
                pass

            # Check nvidia-smi
            result = subprocess.run(
                ['nvidia-smi'],
                capture_output=True,
                text=True
            )
            return result.returncode == 0

        except Exception:
            return False

    # ========================================
    # LINEAR ALGEBRA OPERATIONS
    # ========================================

    def matrix_multiply_gpu(
        self,
        A: np.ndarray,
        B: np.ndarray
    ) -> np.ndarray:
        """
        GPU-accelerated matrix multiplication

        Args:
            A: Matrix A (m x n)
            B: Matrix B (n x p)

        Returns:
            Result matrix (m x p)
        """
        try:
            import torch

            if self.cuda_available:
                # Convert to GPU tensors
                A_gpu = torch.from_numpy(A).cuda()
                B_gpu = torch.from_numpy(B).cuda()

                # Multiply on GPU
                C_gpu = torch.matmul(A_gpu, B_gpu)

                # Return as numpy
                return C_gpu.cpu().numpy()
            else:
                # Fallback to CPU
                return np.matmul(A, B)

        except ImportError:
            # No torch, use numpy
            return np.matmul(A, B)

    def eigenvalues_gpu(
        self,
        matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        GPU-accelerated eigenvalue decomposition

        Args:
            matrix: Square matrix

        Returns:
            (eigenvalues, eigenvectors)
        """
        try:
            import torch

            if self.cuda_available:
                M_gpu = torch.from_numpy(matrix).cuda()
                eigenvalues, eigenvectors = torch.linalg.eig(M_gpu)
                return eigenvalues.cpu().numpy(), eigenvectors.cpu().numpy()
            else:
                return np.linalg.eig(matrix)

        except ImportError:
            return np.linalg.eig(matrix)

    def svd_gpu(
        self,
        matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        GPU-accelerated Singular Value Decomposition

        Args:
            matrix: Input matrix

        Returns:
            (U, S, V) where matrix = U @ diag(S) @ V
        """
        try:
            import torch

            if self.cuda_available:
                M_gpu = torch.from_numpy(matrix).cuda()
                U, S, V = torch.linalg.svd(M_gpu)
                return U.cpu().numpy(), S.cpu().numpy(), V.cpu().numpy()
            else:
                return np.linalg.svd(matrix)

        except ImportError:
            return np.linalg.svd(matrix)

    # ========================================
    # STATISTICAL FUNCTIONS
    # ========================================

    def covariance_matrix_gpu(
        self,
        data: np.ndarray
    ) -> np.ndarray:
        """
        GPU-accelerated covariance matrix calculation

        Args:
            data: Data matrix (n_samples x n_features)

        Returns:
            Covariance matrix (n_features x n_features)
        """
        try:
            import torch

            if self.cuda_available:
                data_gpu = torch.from_numpy(data).cuda()
                # Center the data
                data_centered = data_gpu - data_gpu.mean(dim=0)
                # Covariance
                cov = (data_centered.T @ data_centered) / (data_gpu.shape[0] - 1)
                return cov.cpu().numpy()
            else:
                return np.cov(data.T)

        except ImportError:
            return np.cov(data.T)

    def correlation_matrix_gpu(
        self,
        data: np.ndarray
    ) -> np.ndarray:
        """
        GPU-accelerated correlation matrix

        Args:
            data: Data matrix (n_samples x n_features)

        Returns:
            Correlation matrix
        """
        try:
            import torch

            if self.cuda_available:
                data_gpu = torch.from_numpy(data).cuda()
                return torch.corrcoef(data_gpu.T).cpu().numpy()
            else:
                return np.corrcoef(data.T)

        except ImportError:
            return np.corrcoef(data.T)

    def moving_average_gpu(
        self,
        data: np.ndarray,
        window: int
    ) -> np.ndarray:
        """
        GPU-accelerated moving average

        Args:
            data: Time series data
            window: Window size

        Returns:
            Moving average
        """
        try:
            import torch

            if self.cuda_available:
                data_gpu = torch.from_numpy(data).cuda()
                # Use convolution for moving average
                kernel = torch.ones(window, device='cuda') / window
                result = torch.nn.functional.conv1d(
                    data_gpu.unsqueeze(0).unsqueeze(0),
                    kernel.unsqueeze(0).unsqueeze(0),
                    padding=window // 2
                )
                return result.squeeze().cpu().numpy()
            else:
                return np.convolve(data, np.ones(window)/window, mode='same')

        except ImportError:
            return np.convolve(data, np.ones(window)/window, mode='same')

    # ========================================
    # QUANTUM MATHEMATICS
    # ========================================

    def quantum_state_vector(
        self,
        num_qubits: int,
        amplitudes: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Create quantum state vector

        Args:
            num_qubits: Number of qubits
            amplitudes: State amplitudes (optional, defaults to |0...0>)

        Returns:
            Normalized quantum state vector
        """
        dim = 2 ** num_qubits

        if amplitudes is None:
            # Default to |0...0> state
            state = np.zeros(dim, dtype=complex)
            state[0] = 1.0
        else:
            state = amplitudes.astype(complex)

        # Normalize
        norm = np.sqrt(np.sum(np.abs(state) ** 2))
        if norm > 0:
            state = state / norm

        return state

    def quantum_gate_application(
        self,
        state: np.ndarray,
        gate_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Apply quantum gate to state

        Args:
            state: Quantum state vector
            gate_matrix: Unitary gate matrix

        Returns:
            New quantum state
        """
        return self.matrix_multiply_gpu(gate_matrix, state.reshape(-1, 1)).flatten()

    def quantum_entanglement_entropy(
        self,
        state: np.ndarray,
        subsystem_qubits: int
    ) -> float:
        """
        Calculate entanglement entropy of a subsystem

        Args:
            state: Quantum state vector
            subsystem_qubits: Number of qubits in subsystem

        Returns:
            Von Neumann entropy
        """
        n_qubits = int(np.log2(len(state)))
        dim_A = 2 ** subsystem_qubits
        dim_B = 2 ** (n_qubits - subsystem_qubits)

        # Reshape state as matrix
        psi_matrix = state.reshape(dim_A, dim_B)

        # Compute reduced density matrix
        rho_A = psi_matrix @ psi_matrix.conj().T

        # Eigenvalues
        eigenvalues = np.linalg.eigvalsh(rho_A)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]

        # Von Neumann entropy
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues))

        return float(entropy)

    # ========================================
    # TRADING MATHEMATICS
    # ========================================

    def sharpe_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.02
    ) -> float:
        """
        Calculate Sharpe ratio

        Args:
            returns: Array of returns
            risk_free_rate: Risk-free rate (annual)

        Returns:
            Sharpe ratio
        """
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return float(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252))

    def maximum_drawdown(
        self,
        equity_curve: np.ndarray
    ) -> float:
        """
        Calculate maximum drawdown

        Args:
            equity_curve: Equity curve over time

        Returns:
            Maximum drawdown (negative value)
        """
        cumulative_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - cumulative_max) / cumulative_max
        return float(np.min(drawdown))

    def kelly_criterion(
        self,
        win_probability: float,
        win_loss_ratio: float
    ) -> float:
        """
        Calculate optimal bet size using Kelly Criterion

        Args:
            win_probability: Probability of winning
            win_loss_ratio: Average win / average loss

        Returns:
            Optimal fraction of capital to risk
        """
        if win_probability <= 0 or win_probability >= 1:
            return 0.0

        kelly = (win_probability * win_loss_ratio - (1 - win_probability)) / win_loss_ratio
        return max(0.0, float(kelly))

    def bollinger_bands(
        self,
        prices: np.ndarray,
        window: int = 20,
        num_std: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Bollinger Bands

        Args:
            prices: Price series
            window: Moving average window
            num_std: Number of standard deviations

        Returns:
            (upper_band, middle_band, lower_band)
        """
        middle = self.moving_average_gpu(prices, window)

        # Calculate rolling std
        std = np.array([
            np.std(prices[max(0, i-window):i+1])
            for i in range(len(prices))
        ])

        upper = middle + num_std * std
        lower = middle - num_std * std

        return upper, middle, lower

    def rsi(
        self,
        prices: np.ndarray,
        period: int = 14
    ) -> np.ndarray:
        """
        Calculate Relative Strength Index

        Args:
            prices: Price series
            period: RSI period

        Returns:
            RSI values
        """
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gains = self.moving_average_gpu(gains, period)
        avg_losses = self.moving_average_gpu(losses, period)

        rs = avg_gains / (avg_losses + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        return np.concatenate([[50], rsi])  # Prepend neutral value

    # ========================================
    # NEURAL NETWORK PRIMITIVES
    # ========================================

    def sigmoid_gpu(
        self,
        x: np.ndarray
    ) -> np.ndarray:
        """GPU-accelerated sigmoid activation"""
        try:
            import torch

            if self.cuda_available:
                x_gpu = torch.from_numpy(x).cuda()
                return torch.sigmoid(x_gpu).cpu().numpy()
            else:
                return 1 / (1 + np.exp(-x))

        except ImportError:
            return 1 / (1 + np.exp(-x))

    def relu_gpu(
        self,
        x: np.ndarray
    ) -> np.ndarray:
        """GPU-accelerated ReLU activation"""
        try:
            import torch

            if self.cuda_available:
                x_gpu = torch.from_numpy(x).cuda()
                return torch.relu(x_gpu).cpu().numpy()
            else:
                return np.maximum(0, x)

        except ImportError:
            return np.maximum(0, x)

    def softmax_gpu(
        self,
        x: np.ndarray
    ) -> np.ndarray:
        """GPU-accelerated softmax"""
        try:
            import torch

            if self.cuda_available:
                x_gpu = torch.from_numpy(x).cuda()
                return torch.softmax(x_gpu, dim=-1).cpu().numpy()
            else:
                exp_x = np.exp(x - np.max(x))
                return exp_x / np.sum(exp_x)

        except ImportError:
            exp_x = np.exp(x - np.max(x))
            return exp_x / np.sum(exp_x)

    def batch_normalization_gpu(
        self,
        x: np.ndarray,
        epsilon: float = 1e-5
    ) -> np.ndarray:
        """GPU-accelerated batch normalization"""
        try:
            import torch

            if self.cuda_available:
                x_gpu = torch.from_numpy(x).cuda()
                mean = x_gpu.mean(dim=0)
                var = x_gpu.var(dim=0)
                normalized = (x_gpu - mean) / torch.sqrt(var + epsilon)
                return normalized.cpu().numpy()
            else:
                mean = np.mean(x, axis=0)
                var = np.var(x, axis=0)
                return (x - mean) / np.sqrt(var + epsilon)

        except ImportError:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            return (x - mean) / np.sqrt(var + epsilon)

    def attention_mechanism_gpu(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray
    ) -> np.ndarray:
        """
        GPU-accelerated attention mechanism

        Args:
            query: Query matrix
            key: Key matrix
            value: Value matrix

        Returns:
            Attention output
        """
        # Scaled dot-product attention
        scores = self.matrix_multiply_gpu(query, key.T)
        scores = scores / np.sqrt(key.shape[-1])

        # Softmax
        attention_weights = self.softmax_gpu(scores)

        # Apply to values
        output = self.matrix_multiply_gpu(attention_weights, value)

        return output


# Global instance
_cuda_math = None

def get_cuda_mathematics() -> ALCALACUDAMathematics:
    """Get or create ALCALA CUDA mathematics instance"""
    global _cuda_math
    if _cuda_math is None:
        _cuda_math = ALCALACUDAMathematics()
    return _cuda_math


if __name__ == "__main__":
    # Test CUDA mathematics module
    print("Testing ALCALA CUDA Mathematics Module...")
    print()

    cuda_math = ALCALACUDAMathematics()

    # Test 1: Matrix multiplication
    print("Test 1: Matrix Multiplication")
    A = np.random.randn(100, 100)
    B = np.random.randn(100, 100)
    C = cuda_math.matrix_multiply_gpu(A, B)
    print(f"   Result shape: {C.shape}")
    print()

    # Test 2: Statistical functions
    print("Test 2: Statistical Functions")
    data = np.random.randn(1000, 10)
    cov = cuda_math.covariance_matrix_gpu(data)
    print(f"   Covariance matrix shape: {cov.shape}")
    print()

    # Test 3: Trading indicators
    print("Test 3: Trading Indicators")
    prices = np.cumsum(np.random.randn(100)) + 100
    upper, middle, lower = cuda_math.bollinger_bands(prices)
    print(f"   Bollinger Bands calculated: {len(upper)} points")
    print()

    # Test 4: Quantum mathematics
    print("Test 4: Quantum Mathematics")
    state = cuda_math.quantum_state_vector(3)
    print(f"   3-qubit state vector: {state.shape}")
    print()

    # Test 5: Neural network primitives
    print("Test 5: Neural Network Primitives")
    x = np.random.randn(100, 50)
    activated = cuda_math.relu_gpu(x)
    print(f"   ReLU activation applied: {activated.shape}")
    print()

    print("ALCALA CUDA Mathematics Module Test Complete!")
