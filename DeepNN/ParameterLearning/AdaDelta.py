from ParameterLearning.ParameterLearning import ParameterLearning
import numpy as np

class AdaDelta(ParameterLearning):
    def __init__(self,learningRate=0.005,rho=0.25,epsilon=0.001):
        super().__init__()
        self.lrate=learningRate
        self.rho=rho
        self.eps=epsilon
        
        self.init=True
        
    def UpdateParameter(self,W,B,dW,dB):
        if self.init:
            self.A_dW=np.zeros(W.shape)
            self.A_dB=np.zeros(B.shape)
            self.delta_W=np.zeros(W.shape)
            self.delta_B=np.zeros(B.shape)
            self.init=False
        
        self.A_dW = self.rho*self.A_dW + (1-self.rho)*np.square(dW)
        self.A_dB = self.rho*self.A_dB + (1-self.rho)*np.square(dB)
        
        self.delta_W = self.rho*self.delta_W +(1-self.rho)*np.square(self.lrate*dW)/(self.A_dW+self.eps)
        self.delta_B = self.rho*self.delta_B +(1-self.rho)*np.square(self.lrate*dB)/(self.A_dB+self.eps)
        
        W -= dW*np.sqrt(self.delta_W/(self.A_dW+self.eps))
        B -= dB*np.sqrt(self.delta_B/(self.A_dB+self.eps))

        return W,B
