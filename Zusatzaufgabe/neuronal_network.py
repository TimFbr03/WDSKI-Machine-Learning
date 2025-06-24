import numpy as np
from functions import Functions as fn

class NeuronalNetwork:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, lr: float, seed: int=42):
        np.random.seed(seed)
        self.lr = lr

        # Eingabeschicht
        self.W1 = np.random.randn(hidden_dim, input_dim) * 0.1
        self.b1 = np.zeros((hidden_dim, 1))

        # Ausgabeschicht
        self.W2 = np.random.randn(output_dim, hidden_dim) * 0.1
        self.b2 = np.zeros((output_dim, 1))

    def forward(self, X: np.ndarray) -> np.ndarray | float:
        """
        Performs the forward pass through the neural network.
        - Dot product of the input Weight and X.T + input bias
        - Input acctivation using ReLU function
        - Dot product of output Weight and activated input weight
        - Output activation using Sigmoid function

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features),
                            where n_features should match the input_dim.

        Returns:
            np.ndarray: Output of the network (predicted probabilities)
                        with shape (1, n_samples).
        """

        self.Z1 = np.dot(self.W1, X.T) + self.b1
        self.A1 = fn.relu(self.Z1)

        self.Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = fn.sigmoid(self.Z2)
        return self.A2

    def backward(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Performs the backward pass and updates the network weights using gradient descent.

        Args:
            X (np.ndarray): Input data of shape (n_samples, input_dim).
            y (np.ndarray): True labels of shape (n_samples,) or (n_samples, 1).
                            Values must be binary (0 or 1).

        Returns:
            None
        """

        m = X.shape[0]
        y = y.reshape(1, -1)

        # Error at output
        dZ2 = self.A2 - y
        dW2 = (1 / m) * np.dot(dZ2, self.A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        # Error in the hidden layer
        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = dA1 * fn.relu(self.Z1, True)

        # Gradient in the hidden layer
        dW1 = (1 / m) * np.dot(dZ1, X)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        # Gradient Descent with alpha = lr
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2


    def train(self, X: np.ndarray, y: np.ndarray, epochs: int, verbose: bool=True) -> None:
        '''
        Fits the Neuronal Network onto the Dataset by Computin
        - forward pass,
        - error (mean_squared_error),
        - backward pass,
        for each iteration 'epochs'
        Prints out Epoch and Loss

        Args:
            X: Numpy Array containing the Coordinates of the Points in the Dataset
            y: Numpy Array differentiating the classes with 0 and 1
            epochs: Amount of Training epochs to fit the Dataset
            verbose: Bool to print Progess of the Training
        
        Returns:
            None
        '''

        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = fn.mean_squared_error(y, y_pred)
            self.backward(X, y)

            if verbose and epoch % 100 == 0:
                print(f"{((epoch/epochs)*100):.2f}%, Epoch {epoch}, Loss: {loss:.4f}", end='\r')

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
            Creates a prediction on a datapoint which class it could be

            Args:
                X: np.array with coordinates in the space

            Returns:
                Prediction on the Class
        '''
        y_pred = self.forward(X)
        y_pred = np.array(y_pred)
        return (y_pred > 0.5).astype(int).flatten()

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        '''
        Evaluates the accuracy of the predictions

        Args:
            X: np.ndarray with the coordinates of the training data
            y: np.ndarray containing the class of the training points

        Returns:

        '''
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def info(self):
        """
        Prints out the weights and bias of the Layers
        """
        np.set_printoptions(precision=4, suppress=True)

        W1 = self.W1
        b1 = self.b1.flatten()
        W2_T = self.W2.T
        b2 = self.b2.flatten()

        rows = max(len(W1), len(b1), len(W2_T), len(b2))

        # Column headers
        print(f"Model Parameters:")
        print("-" * 120)
        print(f"{'Input Weights (W1)':<30} {'Bias (b1)':<30} {'Output Weights (W2.T)':<30} {'Bias (b2)':<30}")
        print("-" * 120)

        for i in range(rows):
            # W1
            w1_str = " ".join(f"{x:7.4f}" for x in W1[i]) if i < len(W1) else ""
            # b1
            b1_str = f"{b1[i]:7.4f}" if i < len(b1) else ""
            # W2.T
            w2_str = " ".join(f"{x:7.4f}" for x in W2_T[i]) if i < len(W2_T) else ""
            # b2
            b2_str = f"{b2[i]:7.4f}" if i < len(b2) else ""

            print(f"{w1_str:<30} {b1_str:<30} {w2_str:<30} {b2_str:<30}")



