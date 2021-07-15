import numpy as np

from LossFunction.LossFunction import LossFunction
class CrossEntropy(LossFunction):
    def __init__(self): super().__init__()

    @staticmethod
    def Loss(output,y): return np.sum(-(y*np.log(output)+(1-y)*np.log(1-output)))

    @staticmethod
    def Backward(output,y): return -(y/output - (1-y)/(1-output))