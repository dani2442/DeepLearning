class Layer(object):
    def __init__(self,in_size,out_size,ActivationFunction):
        self.ActivationFunction=ActivationFunction
        self.in_size=in_size
        self.out_size=out_size

    def Forward(self,x): raise Exception("Layer, Forward: Function not implemented")

    def Backward(self): raise Exception("Layer, Backward: Function not implemented")

    def SetLearningMethod(self,lm): self.learningmethod=lm

    def UpdateParameters(self):
        self.W,self.B=self.learningmethod.UpdateParameter(self.W,self.B,self.dW,self.dB)

from Layer.Plain import *