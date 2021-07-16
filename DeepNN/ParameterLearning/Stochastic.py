from ParameterLearning.ParameterLearning import ParameterLearning
import numpy as np

class Stochastic(ParameterLearning):
    def __init__(self,learningRate=0.01,regRate=0): 
        super().__init__(regRate)
        self.lrate=learningRate

    def UpdateParameter(self,W,B,dW,dB):
        W-=self.lrate*(self.regRate*W+dW)
        B-=self.lrate*(self.regRate*B+dB)

        return W,B