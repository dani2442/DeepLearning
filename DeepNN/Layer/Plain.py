from Layer import Layer
from ActivationFunction.Sigmoid import Sigmoid
import numpy as np

class Plain(Layer):
    def __init__(self,in_size,out_size): 
        super().__init__(in_size,out_size,ActivationFunction=Sigmoid)

        self.W=np.random.randn(out_size,in_size)
        self.B=np.random.randn(out_size,1)

        self.dW=np.zeros((out_size,in_size))
        self.dB=np.zeros((out_size,1))

        self.A=np.zeros((out_size,1))
        self.O=np.zeros((out_size,1))

        self.ActivationFunction=Sigmoid

    def Forward(self,x): 
        self.A=np.dot(self.W,x)+self.B
        self.O=self.ActivationFunction.Forward(self.A)
        return self.O
    
    def Backward(self): pass

    def UpdateParameters(self): pass