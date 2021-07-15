from ParameterLearning.ParameterLearning import ParameterLearning
import numpy as np

class Nesterov(ParameterLearning):
    def __init__(self,learningRate=0.01,beta=0.01):
        super().__init__()
        self.beta=beta
        self.lrate=learningRate

        self.init=True
        
    def UpdateParameter(self,W,B,dW,dB):  
        if self.init:
            self.V_dW = np.zeros(W.shape)
            self.V_dB = np.zeros(B.shape)

            self.W2=np.copy(W)
            self.B2=np.copy(B)

            self.init = False
        
        self.hW=np.copy(W)
        self.hB=np.copy(B)

        self.V_dW = self.beta*self.V_dW-self.lrate*dW
        self.V_dB = self.beta*self.V_dB-self.lrate*dB

        W =self.W2+self.V_dW
        B =self.B2+self.V_dB

        self.W2=self.hW
        self.B2=self.hB

        return W,B

        