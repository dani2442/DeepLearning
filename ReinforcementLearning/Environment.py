from typing import List
class Environment(object):
    def __init__(self,state):
        self.state=state
        
    def NextTimeStep(self,action): raise Exception("Function not implemented")
