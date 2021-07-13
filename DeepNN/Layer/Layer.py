class Layer(object):
    def Forward(self,x): raise Exception("Layer, Forward: Function not implemented")

    def Backward(self): raise Exception("Layer, Backward: Function not implemented")

from Layer.Plain import *