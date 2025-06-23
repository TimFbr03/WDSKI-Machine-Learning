import numpy as np

class Functions:
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
    def mean_squared_error(y_true: np.ndarray | float, y_pred: np.ndarray | float) -> np.float64:
        '''
        Calculates the mean squared error 

        Args:
            y_true:
            y_pred:
        Returns:
            mse:
        '''
        mean_squared_error = np.mean((y_true - y_pred) ** 2)
        return mean_squared_error