from Initializers.Initializer import Initializer
import numpy as np
class Xavier(Initializer):
    def __init__(self): 
        super().__init__()
    
    def Init(self,size):
        return Initializer.rng.normal(0,1/size[1],size)
