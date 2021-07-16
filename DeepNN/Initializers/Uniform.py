from Initializers.Initializer import Initializer
import numpy as np
class Uniform(Initializer):
    def __init__(self,a=-1,b=1): 
        super().__init__()
        self.a=a
        self.b=b
    
    def Init(self,size):
        return Initializer.rng.normal(self.mu,self.sigma,size)
