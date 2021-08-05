import numpy as np
from Agent import Agent
class BanditAgent(Agent):
	def __init__(self,n):
		self.n_actions=n

		self.N=np.zeros((n))
		self.Q=np.zeros((n))

		self.action=None
		self.eps=0.1


	# e-greedy selection
	def Action(self,state,reward):
		if reward!=None:
			self.N[self.action]+=1
			self.Q[self.action]=(reward+(self.N[self.action]-1)*self.Q[self.action])/self.N[self.action]

		if np.random.uniform()<self.eps:
			self.action=np.random.randint(0,self.n_actions)
		else:
			self.action=np.argmax(self.Q)

		return self.action

		
