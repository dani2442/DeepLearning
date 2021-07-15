import numpy as np
from ActivationFunction.ActivationFunction import ActivationFunction
class Tanh(ActivationFunction):
    def __init__(self): super().__init__()

    @staticmethod
    def Forward(x): 
        return np.tanh(x)

    @staticmethod
    def Backward(output,a,dO): 
        return 1-np.square(output)