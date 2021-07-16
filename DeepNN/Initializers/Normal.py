from Initializers.Initializer import Initializer
import numpy as np
class Normal(Initializer):
    def __init__(self,mu=0,sigma=1): 
        super().__init__()
        self.mu=mu
        self.sigma=sigma
    
    def Init(self,size):
        return Initializer.rng.normal(self.mu,self.sigma,size)
