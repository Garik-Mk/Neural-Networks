import numpy as np
import activation_functions

class Neural_Net(object):
    def __init__(self) -> None:
        pass

class Layer(object):
    def _init__(self) -> None:
        self.weights = [[]]
        self.bias = []
        self.size = 0
        self.activation_function = None

    def activation(self, value: float) -> float:
        """This function calculates value of activation function."""
        res = 0
        func = self.activation_function
        exec('res = activation_functions.' + func + '({})'.format(value))
        return res
        
    def forward(self, input_vector: list, weights: list, bias: float) -> float:
        res = []
        for i in range(self.size):
            temp = np.dot(input_vector, weights[i])
            temp += bias[i]
            temp = self.activation(temp)
            res.append(temp)
        return res
   

