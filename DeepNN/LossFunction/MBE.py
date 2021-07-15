import numpy as np
from LossFunction.LossFunction import LossFunction

# Only for Sigmoid in previous layer
class MBE(LossFunction):
    def __init__(self): super().__init__()

    @staticmethod
    def Loss(output,y): return np.sum(y-output)/len(output)

    @staticmethod
    def Backward(output,y): return -1/len(output)
