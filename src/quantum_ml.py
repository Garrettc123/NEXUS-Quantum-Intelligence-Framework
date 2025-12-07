"""Quantum Machine Learning Module

Implements quantum neural networks and quantum-enhanced ML algorithms.
"""

import numpy as np
from typing import List, Tuple, Optional
import logging
from src.quantum_core import QuantumProcessor

logger = logging.getLogger(__name__)


class QuantumNeuralNetwork:
    """Quantum Neural Network with variational quantum circuits"""
    
    def __init__(self, n_qubits: int, n_layers: int):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.processor = QuantumProcessor()
        
        # Initialize parameters
        self.params = np.random.randn(n_layers, n_qubits, 3) * 0.1
        
    def forward(self, x: np.ndarray) -> float:
        """Forward pass through quantum circuit"""
        self.processor.initialize(self.n_qubits)
        
        # Encode input
        for i in range(min(len(x), self.n_qubits)):
            self.processor.apply_rotation(i, x[i], 'y')
            
        # Variational layers
        for layer in range(self.n_layers):
            # Rotation gates
            for i in range(self.n_qubits):
                self.processor.apply_rotation(i, self.params[layer, i, 0], 'x')
                self.processor.apply_rotation(i, self.params[layer, i, 1], 'y')
                self.processor.apply_rotation(i, self.params[layer, i, 2], 'z')
                
            # Entangling gates
            for i in range(self.n_qubits - 1):
                self.processor.apply_cnot(i, i + 1)
                
        # Measure expectation value
        state = self.processor.get_state_vector()
        expectation = np.real(np.vdot(state, state))
        
        return expectation
        
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, lr: float = 0.01):
        """Train QNN using parameter shift rule"""
        for epoch in range(epochs):
            total_loss = 0
            
            for xi, yi in zip(X, y):
                # Forward pass
                pred = self.forward(xi)
                loss = (pred - yi) ** 2
                total_loss += loss
                
                # Compute gradients using parameter shift
                for layer in range(self.n_layers):
                    for qubit in range(self.n_qubits):
                        for param_idx in range(3):
                            # Shift parameter
                            self.params[layer, qubit, param_idx] += np.pi / 2
                            plus = self.forward(xi)
                            
                            self.params[layer, qubit, param_idx] -= np.pi
                            minus = self.forward(xi)
                            
                            # Restore parameter
                            self.params[layer, qubit, param_idx] += np.pi / 2
                            
                            # Gradient
                            grad = (plus - minus) / 2
                            
                            # Update
                            self.params[layer, qubit, param_idx] -= lr * grad * (pred - yi)
                            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {total_loss / len(X):.4f}")


class QuantumKernelClassifier:
    """Quantum kernel methods for classification"""
    
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.processor = QuantumProcessor()
        self.support_vectors = None
        self.support_labels = None
        
    def quantum_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute quantum kernel between two data points"""
        self.processor.initialize(self.n_qubits)
        
        # Encode x1
        for i in range(min(len(x1), self.n_qubits)):
            self.processor.apply_rotation(i, x1[i], 'y')
            
        # Encode x2 (inverted)
        for i in range(min(len(x2), self.n_qubits)):
            self.processor.apply_rotation(i, -x2[i], 'y')
            
        # Measure overlap
        state = self.processor.get_state_vector()
        kernel_value = np.abs(state[0]) ** 2
        
        return kernel_value
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit quantum kernel classifier"""
        self.support_vectors = X
        self.support_labels = y
        logger.info(f"Trained on {len(X)} samples")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using quantum kernel"""
        predictions = []
        
        for x in X:
            # Compute kernel with all support vectors
            kernels = [self.quantum_kernel(x, sv) for sv in self.support_vectors]
            
            # Weighted vote
            vote = sum(k * y for k, y in zip(kernels, self.support_labels))
            predictions.append(1 if vote > 0 else -1)
            
        return np.array(predictions)


class QuantumBoltzmannMachine:
    """Quantum Boltzmann Machine for unsupervised learning"""
    
    def __init__(self, n_visible: int, n_hidden: int):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_qubits = n_visible + n_hidden
        self.processor = QuantumProcessor()
        
        # Initialize weights
        self.weights = np.random.randn(n_visible, n_hidden) * 0.1
        
    def sample(self, n_samples: int = 100) -> np.ndarray:
        """Generate samples from quantum Boltzmann distribution"""
        samples = []
        
        for _ in range(n_samples):
            self.processor.initialize(self.n_qubits)
            
            # Create thermal state
            for i in range(self.n_qubits):
                self.processor.apply_hadamard(i)
                
            # Apply interactions
            for i in range(self.n_visible):
                for j in range(self.n_hidden):
                    angle = self.weights[i, j]
                    self.processor.apply_rotation(self.n_visible + j, angle, 'z')
                    self.processor.apply_cnot(i, self.n_visible + j)
                    
            # Measure
            sample = self.processor.measure()
            samples.append(sample[:self.n_visible])
            
        return np.array(samples)
        
    def train(self, data: np.ndarray, epochs: int = 10, lr: float = 0.01):
        """Train using contrastive divergence"""
        for epoch in range(epochs):
            for sample in data:
                # Positive phase
                positive_grad = np.outer(sample, np.random.randn(self.n_hidden))
                
                # Negative phase
                generated = self.sample(n_samples=1)[0]
                negative_grad = np.outer(generated, np.random.randn(self.n_hidden))
                
                # Update weights
                self.weights += lr * (positive_grad - negative_grad)
                
            logger.info(f"QBM Epoch {epoch + 1}/{epochs} complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*60)
    print("Quantum Machine Learning Demo")
    print("="*60)
    
    # Quantum Neural Network
    print("\n1. Quantum Neural Network:")
    qnn = QuantumNeuralNetwork(n_qubits=4, n_layers=2)
    X_train = np.random.randn(10, 4)
    y_train = np.random.randn(10)
    print("   Training QNN...")
    qnn.train(X_train, y_train, epochs=5)
    
    # Quantum Kernel Classification
    print("\n2. Quantum Kernel Classifier:")
    qkc = QuantumKernelClassifier(n_qubits=4)
    y_class = np.random.choice([-1, 1], 10)
    qkc.fit(X_train, y_class)
    predictions = qkc.predict(X_train[:3])
    print(f"   Predictions: {predictions}")
    
    print("\n" + "="*60)
