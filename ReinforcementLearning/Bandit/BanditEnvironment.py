import numpy as np
from Environment import Environment

class BanditEnvironment(Environment):
    def __init__(self,n): 
        super().__init__(None)
        self.n=n
        self.r=np.random.randn(n) # reward mu values (reward for i action is given by N(r[i],1))

    def NextTimeStep(self, action):
        return None,np.random.normal(self.r[action],scale=1.0)