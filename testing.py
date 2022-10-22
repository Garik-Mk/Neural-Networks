"""Temprory script, used only while testing activation functions"""
from matplotlib import pyplot as plt
import numpy as np
from activation_functions import gelu

x = np.arange(-10,10,0.1)
y = [gelu(i) for i in x]
plt.plot(x, y)
plt.show()