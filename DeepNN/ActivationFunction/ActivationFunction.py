import numpy as np

class ActivationFunction(object):
    def __init__(self): pass
    def Forward(self,x): raise Exception("Function not impolemented")

    def Backward(self,output,a, dO): raise Exception("Function not impolemented")

from ActivationFunction.Sigmoid import *