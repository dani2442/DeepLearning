from ParameterLearning.ParameterLearning import ParameterLearning
import numpy as np

class AdaGrad(ParameterLearning):
    def __init__(self,learningRate=0.01):
        super().__init__()
        self.lrate=learningRate
        self.eps=0.0001
        
        self.init=True
        
    def UpdateParameter(self,W,B,dW,dB):
        if self.init:
            self.A_dW=np.zeros(W.shape)
            self.A_dB=np.zeros(B.shape)
            self.init=False
        
        self.A_dW += np.square(dW)
        self.A_dB += np.square(dB)
        
        W -= (self.lrate*dW)/np.sqrt(self.A_dW+self.eps)
        B -= (self.lrate*dB)/np.sqrt(self.A_dB+self.eps)

        return W,B
