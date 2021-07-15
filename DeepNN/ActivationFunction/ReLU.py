import numpy as np
from ActivationFunction.ActivationFunction import ActivationFunction
class ReLU(ActivationFunction):
    def __init__(self): super().__init__()

    @staticmethod
    def Forward(x): 
        return np.maximum(x,0)

    @staticmethod
    def Backward(output,a,dO): 
        back=np.copy(output)
        back[back<=0]=0
        back[back>0]=1
        return back*dO