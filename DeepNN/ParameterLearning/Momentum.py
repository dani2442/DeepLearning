from ParameterLearning.ParameterLearning import ParameterLearning
import numpy as np

class Momentum(ParameterLearning):
    def __init__(self,learningRate=0.01,momentumParam=0.1,regRate=0):
        super().__init__(regRate)
        self.lrate=learningRate
        self.mparam=momentumParam

        self.init=True
        
    def UpdateParameter(self,W,B,dW,dB):  
        if self.init:
            self.V_dW = np.zeros(W.shape)
            self.V_dB = np.zeros(B.shape)
            self.init = False

        self.V_dW = self.V_dW*self.mparam-self.lrate*dW
        self.V_dB = self.V_dB*self.mparam-self.lrate*dB
        
        W += self.V_dW - self.lrate*self.regRate*W
        B += self.V_dB - self.lrate*self.regRate*B

        return W,B
        
