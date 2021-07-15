import numpy as np

class ActivationFunction(object):
    @staticmethod
    def Forward(x): raise Exception("Function not impolemented")

    @staticmethod
    def Backward(output,a, dO): raise Exception("Function not impolemented")

from ActivationFunction.Sigmoid import *