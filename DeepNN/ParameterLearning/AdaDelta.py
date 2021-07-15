from ParameterLearning.ParameterLearning import ParameterLearning
import numpy as np

class AdaDelta(ParameterLearning):
    def __init__(self,learningRate=0.01,rho=0.5,epsilon=0.001):
        super().__init__()
        self.lrate=learningRate
        self.rho=rho
        self.eps=epsilon
        
        self.init=True
        
    def UpdateParameter(self,W,B,dW,dB):
        if self.init:
            self.A_dW=np.zeros(W.shape)
            self.A_dB=np.zeros(B.shape)
            self.delta_W=0
            self.delta_B=0
            self.init=False
        
        self.A_dW = self.rho*self.A_dW + (1-self.rho)*np.square(dW)
        self.A_dB = self.rho*self.A_dB + (1-self.rho)*np.square(dB)
        
        self.delta_W = self.rho*self.delta_W +(1-self.rho)*np.square((self.lrate*dW)/np.sqrt(self.A_dW))
        self.delta_B = self.rho*self.delta_B +(1-self.rho)*np.square((self.lrate*dB)/np.sqrt(self.A_dB))
        
        W -= dW/np.sqrt(self.delta_W/self.A_dW + self.eps)
        B -= dB/np.sqrt(self.delta_B/self.A_dB + self.eps)

        return W,B
