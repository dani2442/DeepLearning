from ParameterLearning.ParameterLearning import ParameterLearning
import numpy as np

class AdaGrad(ParameterLearning):
    def __init__(self,learningRate=0.01,epsilon=0.001):
        super().__init__()
        self.lrate=learningRate
        self.eps=epsilon
        
    def UpdateParameter(self,W,B,dW,dB):
        
        A_dW=np.zeros(W.shape)
        A_dB=np.zeros(B.shape)
        
        A_dW += dW**2
        A_dB += dB**2
        
        W -= (self.lrate*dW)/np.sqrt(A_dW-self.eps)
        B -= (self.lrate*dB)/np.sqrt(A_dB-self.eps)
        
        # los valores de A_dW y A_dB deben inicializarse en la 
        # primera iteración y guardarse después de cada iteración