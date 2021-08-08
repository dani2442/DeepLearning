"""
from Agent import Agent
import numpy as np
class BoardEscapeAgent(Agent):
    def __init__(self,P):
        super().__init__()
        self.P=P

    def Action(self,state,reward): 
        return np.argmax

    def GetProbPolicy(self,action,state=None): return self.P[action]
"""