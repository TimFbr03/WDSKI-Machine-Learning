import numpy as np

class Functions:
    @staticmethod
    def linear(x: np.ndarray) -> np.ndarray:
        return x

    @staticmethod
    def sigmoid(x: np.ndarray, derivative: bool=False) -> np.ndarray | float:
        '''
        Applies the sigmoid function element-wise to the input.

        Args:
            x: A scalar or NumPy array
            derivative: A bool to define if the function itself or the derivative is to be used
        Returns:
            Sigmoid activation of the input
        '''
        sig = 1 / (1 + np.exp(-x))

        if derivative:
            return sig * (1 - sig)

        return sig
    
    @staticmethod
    def relu(x: np.ndarray, derivative=False) -> np.ndarray:
        '''
        Applies the Rectified Linear Unit elemtent-wise to the input.

        Args:
            x: A scalar or a NumPy array
            derivative: bool to define if the funtion or dericative is to be used
        Returns:
            ReLU activation of the input
        '''
        if derivative:
            return (x > 0).astype(float)
        return np.maximum(0, x)
    
    @staticmethod
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def binary_cross_entropy(W1: np.ndarray, W2: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, lambda_: float=0.01) -> float:
        '''
        Computes binary cross-entropy loss with L2 regularization.

        Args:
            y_true: Ground truth labels (0 or 1)
            y_pred: Predicted probabilities (between 0 and 1)
            W1, W2: Weight matrices to regularize
            lambda_: Regularization strength
        Returns:
            Scalar loss value
        '''
        eps = 1e-8
        base_loss = -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))
        reg_loss = (lambda_ / 2) * (np.sum(W1**2) + np.sum(W2**2))
        return base_loss + reg_loss # type: ignore