from Layer.Layer import Layer
from ActivationFunction.ActivationFunction import *
import numpy as np
class Plain(Layer):
    def __init__(self,in_size,out_size,ActivationFunction=Sigmoid): 
        super().__init__(in_size,out_size,ActivationFunction)

        self.dW=np.zeros((out_size,in_size))
        self.dB=np.zeros((out_size,1))

        self.A=np.zeros((out_size,1))
        self.O=np.zeros((out_size,1))

        self.dIn=np.zeros((in_size,1))

    def Init(self,initializer):
        self.W=initializer.Init((self.out_size,self.in_size))
        self.B=initializer.Init((self.out_size,1))

    def Forward(self,x): 
        self.A=np.dot(self.W,x)+self.B
        self.O=self.ActivationFunction.Forward(self.A)
        return self.O
    
    def Backward(self,O,dO): # O: output of previous layer  dO: derivative of next layer
        dA=self.ActivationFunction.Backward(self.O,self.A,dO)
        self.dW=np.dot(dA,O.T)
        self.db=np.sum(dA,axis=1,keepdims = True)
        self.dIn=np.dot(self.W.T,dA)
        return self.dIn


