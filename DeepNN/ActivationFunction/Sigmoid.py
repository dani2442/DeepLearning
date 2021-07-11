from ActivationFunction import ActivationFunction
import numpy as np


class Sigmoid(ActivationFunction):
    def __init__(self): super().__init__()

    def Forward(self,x): return 1/(1+np.exp(-x))

    def Backward(self,output,a): return output*(1-output)