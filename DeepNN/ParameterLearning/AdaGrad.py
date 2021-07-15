from ParameterLearning.ParameterLearning import ParameterLearning
import numpy as np

class AdaGrad(ParameterLearning):
    def __init__(self,learningRate=0.01,epsilon=0.001):
        super().__init__()
        self.lrate=learningRate
        self.eps=epsilon
        
        self.init=True
        
    def UpdateParameter(self,W,B,dW,dB):
        if self.init:
            self.A_dW=np.zeros(W.shape)
            self.A_dB=np.zeros(B.shape)
            self.init=False
        
        self.A_dW += dW**2
        self.A_dB += dB**2
        
        W -= (self.lrate*dW)/np.sqrt(self.A_dW-self.eps)
        B -= (self.lrate*dB)/np.sqrt(self.A_dB-self.eps)