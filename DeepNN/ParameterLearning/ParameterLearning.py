class ParameterLearning(object):
    def UpdateParameter(self,W,B,dW,dB): raise Exception("ParameterLeraning: Function not implemented")

from ParameterLearning.Stochastic import *
from ParameterLearning.Momentum import *
from ParameterLearning.AdaGrad import *
from ParameterLearning.RMSProp import *
from ParameterLearning.Adam import *
from ParameterLearning.AdaDelta import *
from ParameterLearning.Nesterov import *