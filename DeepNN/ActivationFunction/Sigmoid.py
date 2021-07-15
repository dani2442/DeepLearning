import numpy as np
from ActivationFunction.ActivationFunction import ActivationFunction
class Sigmoid(ActivationFunction):
    def __init__(self): super().__init__()

    @staticmethod
    def Forward(x): 
        return 1/(1+np.exp(-x))

    @staticmethod
    def Backward(output,a,dO): return output*(1-output)*dO