from ParameterLearning.ParameterLearning import ParameterLearning
import numpy as np

class Adam(ParameterLearning):
    def __init__(self,learningRate=0.01,rho=0.5,rho_f=0.5,epsilon=0.001,regRate=0):
        super().__init__(regRate)
        self.lrate=learningRate
        self.rho=rho
        self.rho_f=rho_f
        self.eps=epsilon
        
        self.init=True

        self._rho_=rho
        self._rho_f_=rho_f
        
    def UpdateParameter(self,W,B,dW,dB):
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
        
        self.lrate *= np.sqrt(1-self._rho_)/(1-self._rho_f_)
        
        W -=self.regRate*self.lrate*W+ (self.lrate*self.F_dW)/np.sqrt(self.A_dW + self.eps)
        B -=self.regRate*self.lrate*B+ (self.lrate*self.F_dB)/np.sqrt(self.A_dB + self.eps)

        self._rho_*=self.rho
        self._rho_f_*=self._rho_f_
        return W,B
