from ParameterLearning.ParameterLearning import ParameterLearning
import numpy as np

class RMSProp(ParameterLearning):
    def __init__(self,learningRate=0.01,rho=0.5,epsilon=0.001,regRate=0):
        super().__init__(regRate)
        self.lrate=learningRate
        self.rho=rho
        self.eps=epsilon
        self.init=True
        
    def UpdateParameter(self,W,B,dW,dB):
        if self.init:
            self.A_dW=np.zeros(W.shape)
            self.A_dB=np.zeros(B.shape)
            self.init=False
        
        self.A_dW = self.rho*self.A_dW + (1-self.rho)*np.square(dW) 
        self.A_dB = self.rho*self.A_dB + (1-self.rho)*np.square(dB)
        
        W -=W*self.regRate*self.lrate+ (self.lrate*dW)/np.sqrt(self.A_dW+self.eps)
        B -=B*self.regRate*self.lrate+ (self.lrate*dB)/np.sqrt(self.A_dB+self.eps)

        return W,B
