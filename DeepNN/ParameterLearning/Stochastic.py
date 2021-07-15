from ParameterLearning.ParameterLearning import ParameterLearning
import numpy as np

class Stochastic(ParameterLearning):
    def __init__(self,learningRate=0.01,rateDecay=0.001): 
        super().__init__()
        self.lrate=learningRate
        self.rdecay=rateDecay

    def UpdateParameter(self,W,B,dW,dB):
        W-=self.lrate*dW
        B-=self.lrate*dB

        return W,B