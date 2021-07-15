import numpy as np

from LossFunction.LossFunction import LossFunction
class MAE(LossFunction):
    def __init__(self): super().__init__()

    @staticmethod
    def Loss(output,y): return np.sum(np.abs(output-y)/len(output))

    @staticmethod
    def Backward(output,y): return (output-y)/(len(output)*np.abs(output-y))