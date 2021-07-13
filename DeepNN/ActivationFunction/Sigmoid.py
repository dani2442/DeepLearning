import numpy as np
from ActivationFunction.ActivationFunction import ActivationFunction

class Sigmoid(ActivationFunction):
    def __init__(self): super().__init__()

    def Forward(self,x): return 1/(1+np.exp(-x))

    def Backward(self,output,a,dO): return output*(1-output)*dO