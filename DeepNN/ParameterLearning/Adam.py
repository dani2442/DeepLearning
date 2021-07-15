from ParameterLearning.ParameterLearning import ParameterLearning
import numpy as np

class Adam(ParameterLearning):
    def __init__(self,learningRate=0.01,rho=0.5,rho_f=0.5,epsilon=0.001):
        super().__init__()
        self.lrate=learningRate
        self.rho=rho
        self.rho_f=rho_f
        self.eps=epsilon
        
        self.init=True
        
    def UpdateParameter(self,W,B,dW,dB,iter):
        if self.init:
            self.A_dW=np.zeros(W.shape)
            self.A_dB=np.zeros(B.shape)
            self.F_dW=np.zeros(W.shape)
            self.F_dB=np.zeros(B.shape)
            self.init=False
        
        self.A_dW = self.rho*self.A_dW + (1-self.rho)*np.square(dW)
        self.A_dB = self.rho*self.A_dB + (1-self.rho)*np.square(dB)
        
        self.F_dW = self.rho_f*self.F_dW + (1-self.rho_f)*dW
        self.F_dB = self.rho_f*self.F_dB + (1-self.rho_f)*dB
        
        self.lrate *= sqrt(1-self.rho**iter)/(1-self.rho_f**iter)
        
        W -= (self.lrate*self.F_dW)/np.sqrt(self.A_dW + self.eps)
        B -= (self.lrate*self.F_dB)/np.sqrt(self.A_dB + self.eps)
