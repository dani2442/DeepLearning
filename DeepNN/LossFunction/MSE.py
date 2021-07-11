import numpy as np
from LossFunction import LossFunction

class MSE(LossFunction):
    def __init__(self): super().__init__()

    def Loss(self,output,y): return np.square(output-y)/2

    def Backward(self,output,y): return output-y