from ParameterLearning import ParameterLearning
import numpy as np

class Momentum(ParameterLearning):
    def __init__(self,learningRate=0.01,momentumParam=0.01):
        super().__init__()
        self.lrate=learningRate
        self.mparam=momentumParam
        
    def UpdateParamter(self,W,B,dW,dB):
        
        V_dW=np.zeros(W.shape)
        V_dB=np.zeros(B.shape)
        
        V_dW-=(self.lrate*dW)/(1-self.mparam)
        V_dB-=(self.lrate*dB)/(1-self.mparam)
        
        W+=V_dW
        B+=V_dB
        
        # los valores de V_dW y V_dB deben inicializarse en la primera
        # iteración y guardarse después de cada iteración