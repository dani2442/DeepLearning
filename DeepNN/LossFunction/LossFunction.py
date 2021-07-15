class LossFunction(object):
    @staticmethod
    def Loss(output,y): raise Exception("Loss: Function not implemented")

    @staticmethod
    def Backward(x,y): raise Exception("Backward: Function not implemented")

from LossFunction.MSE import *