"""
from Environment import Environment
import numpy as np
class BoardEscapeEnvironment(Environment):
	def __init__(self,state,n=5,stop_state=[[0,0],[3,3]],reward=np.full((5,5),-1)):
		super().__init__(np.array(state))
		self.n=n
		self.R=reward
		self.action=np.array([[1,0],[-1,0],[0,1],[0,-1]])
		self.stop=np.array(stop_state)

	def NextTimeStep(self, action_id):
		new_s=self.state+self.action[action_id]
		if np.prod((new_s>=0)*(new_s<self.n)):
			self.state=new_s
		return self.state,-1

	def GetProbStateReward(self,state,action):
		P1=state+self.action[action]==self.action
		P2=None
"""
