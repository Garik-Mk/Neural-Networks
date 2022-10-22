
from math import sqrt
import numpy as np


def binary_step(value: float) -> int:
    """ 
    It cannot provide multi-value outputs—for example, 
    it cannot be used for multi-class classification problems. 
    
    The gradient of the step function is zero, 
    which causes a hindrance in the backpropagation process.
    """
    return 0 if value < 0 else 1

def linear(value: float) -> float:
    """
    The linear activation function, also known as "no activation"
    or "identity function" (multiplied x1.0),is where the activa-
    tion is proportional to the input.

    The function doesn't do anything to the weighted sum of the 
    input, it simply spits out the value it was given.
    """
    return value

def sigmoid(value: float) -> float:
    """
    This function takes any real value as input and outputs 
    values in the range of 0 to 1. 

    The larger the input (more positive), the closer the output
    value will be to 1.0, whereas the smaller the input (more negative),
    the closer the output will be to 0.0, as shown below.
    """
    exponent = np.exp(value)
    return 1 / (1 + exponent)

def dsigmoid(value: float) -> float:
    """Derivative of sigmoid.
    
    The gradient values are only significant for range -3 to 3.

    It implies that for values greater than 3 or less than -3,
    the function will have very small gradients. As the gradient
    value approaches zero, the network ceases to learn and suffers
    from the Vanishing gradient problem.
    """

    return (sigmoid(value) * (1 - sigmoid(value)))

def tanh(value: float) -> float:
    """Hyperbolic Tangent.

    Tanh function is very similar to the sigmoid/logistic activation
    function, and even has the same S-shape with the difference in
    output range of -1 to 1. In Tanh, the larger the input (more positive),
    the closer the output value will be to 1.0, whereas the smaller the
    input (more negative), the closer the output will be to -1.0.
    """
    pos_exponent = np.exp(value) # positive value exponent
    neg_exponent = np.exp(-value) # negative value exponent

    return (pos_exponent - neg_exponent) / (pos_exponent + neg_exponent)

def dtanh(value: float) -> float:
    """Derivative of Hyperbolic Tangent.
    
    It also faces the problem of vanishing gradients similar to the sigmoid
    activation function. Plus the gradient of the tanh function is much steeper
    as compared to the sigmoid function."""

    return 1 - (tanh(value)**2)

def relu(value: float) -> float:
    """ReLU stands for Rectified Linear Unit. 

    Although it gives an impression of a linear function, ReLU has a derivative
    function and allows for backpropagation while simultaneously making it
    computationally efficient.
    """

    return max(0, value)

def drelu(value: float) -> int:
    """Derivative of ReLU
    The negative side of the graph makes the gradient value zero. Due to
    this reason, during the backpropagation process, the weights and biases
    for some neurons are not updated. This can create dead neurons which
    never get activated.
    """
    
    return 1 if value >= 0 else 0

def leaky_relu(value: float) -> float:
    """Leaky ReLU is an improved version of ReLU function to solve the
    Dying ReLU problem as it has a small positive slope in the negative area.
    """

    return max(0.1 * value, value)

def dleaky_relu(value: float) -> float:
    """Derivative of the Leaky ReLU function."""
    return 1 if value >= 0 else 0.01

def param_relu(value: float, alpha: float) -> float:
    """The parameterized ReLU function is used when the leaky ReLU function
    still fails at solving the problem of dead neurons, and the relevant 
    information is not successfully passed to the next layer. """
    return max(alpha*value, value)

def elu(value: float, alpha: float) -> float:
    """Exponential Linear Unit, or ELU for short, is also a variant of ReLU
    that modifies the slope of the negative part of the function. 

    ELU uses a log curve to define the negativ values unlike the leaky ReLU 
    and Parametric ReLU functions with a straight line."""

    if value >= 0:
        return value
    return alpha * (np.exp(value) - 1)

def delu(value: float, alpha: float = 1) -> float:
    """Derivative of Exponential Linear Unit"""
    if value >= 0:
        return value
    return elu(value, alpha) + alpha

def softmax(values: list) -> list:
    """
    Softmax calculates the relative probabilities. Similar to the
    sigmoid/logistic activation function, the SoftMax function returns
    the probability of each class. 

    It is most commonly used as an activation function for the last
    layer of the neural network in the case of multi-class classification.
    """
    exponent_sum = 0
    for value in values:
        exponent_sum += np.exp(value)
    
    res = [np.exp(value)/exponent_sum for value in values]
    return res

def dsoftmax(values: list) -> list:
    """Derivative of softmax."""
    res = []
    for value in values:
        res.append(softmax(value) * (1 - (softmax(value))))
    return res

def swish(value: float) -> float:
    """
    It is a self-gated activation function developed by researchers at Google.

    Swish consistently matches or outperforms ReLU activation function on 
    deep networks applied to various challenging domains such as image classification, 
    machine translation etc. """

    return value * sigmoid(value)

def gelu(value: float) -> float:
    """The Gaussian Error Linear Unit (GELU) activation function
    is compatible with BERT, ROBERTa, ALBERT, and other top NLP models.
    This activation function is motivated by combining properties from
    dropout, zoneout, and ReLUs. 
    """
    res = value + (0.044715 * (value**3))
    res *= sqrt(2/np.pi)
    res = tanh(res)
    res += 1
    res *= 0.5 * value

    return res

def dgelu(value: float) -> float:
    """Derivative of gelu"""
    pass
    #TODO



