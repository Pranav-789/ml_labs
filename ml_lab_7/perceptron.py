import numpy as np
from pydantic import BaseModel

class layer(BaseModel):
    neurons_no: int
    activation_fn: str
    layer_bias: float

def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Sigmoid is a mathematical function that maps any real-valued number into a value between 0 and 1.
    Its characteristic "S"-shaped curve makes it particularly useful in scenarios where we need to convert outputs into probabilities.
    This function is often called the logistic function.
    """
    base = 1 + np.exp(-z)
    return 1/base

def linear(z: np.ndarray)->np.ndarray:
    """
    The linear activation function which returns the output of the weighted sum itself.
    This is to be used for regression type problem statements
    """
    return z

def step(z: np.ndarray)->np.ndarray:
    """
    threshold-based activation function that outputs a binary value based on whether the input exceeds a specific limit.
    """
    return np.where(z >= 0, 1, 0)
    
def relu(z: np.ndarray) -> np.ndarray:
    """
    outputs the input directly if it is positive and returns zero otherwise
    """
    return np.maximum(0, z)

activation_records = {
    'sigmoid': sigmoid,
    'linear': linear,
    'step': step,
    'relu': relu
}

def weightedSum(W: np.ndarray, X: np.ndarray, b: float):
    if W.shape!= X.shape:
        raise ValueError(
            f"Invalid input received.\nThe Expected input dimension is {len(W)}"
        )
    return np.dot(W, X) + b


class Perceptron:
    def __init__(self, dim: tuple, b: float, activation: str = 'linear'):
        if activation not in activation_records:
            available = ", ".join(activation_records.keys())

            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Available activations: {available}"
            )
        self.activate = activation_records[activation]
        self.activation_name = activation
        self.weights = np.random.randn(*dim)*0.01
        self.bias = b

    def __repr__(self):
        w_str = np.array2string(self.weights, precision=4, separator=',')
        return f"Perceptron(weights={w_str}, bias={self.bias}, activation='{self.activation_name}')"

    def get_output(self, X: np.ndarray):
        z = weightedSum(self.weights, X, self.bias)
        return self.activate(z)

    def train(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 1000, lr: float = 0.01):
        for _ in range(epochs):
            for x, y in zip(X_train, y_train):
                delta = self.get_output(x) - y

                self.weights -= lr*delta*x
                self.bias -= lr*delta
        pass