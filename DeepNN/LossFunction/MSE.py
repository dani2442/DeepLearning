import numpy as np

from LossFunction.LossFunction import LossFunction
class MSE(LossFunction):
    def __init__(self): super().__init__()

    @staticmethod
    def Loss(output,y): return np.sum(np.square(output-y)/2)

    @staticmethod
    def Backward(output,y): return output-y