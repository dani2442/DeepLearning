import numpy as np

from LossFunction.LossFunction import LossFunction
class L2Regularization(LossFunction):
    def __init__(self): super().__init__()

    @staticmethod
    def Loss(output,y,W,regParam=0.001): return np.sum(np.square(output-y)/2)/len(output) + regParam*np.sum(np.square(W))

    @staticmethod
    def Backward(output,y,W,regParam=0.001): return (output-y)/len(output) + 2*regParam*W