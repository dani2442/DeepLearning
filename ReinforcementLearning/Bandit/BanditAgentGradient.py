import numpy as np
from Agent import Agent
class BanditAgentGradient(Agent):
	def __init__(self,n,l_rate=0.1):
		self.n_actions=n

		self.H=np.zeros((n)) # numerical preference for each action
		self.P=np.zeros((n)) # Policy
		self.r=0.0
		self.t=0

		self.action=None
		self.l_rate=l_rate

	def Action(self,state,reward):
		if reward!=None:
			self.r+=(reward-self.r)/self.t

			temp_h=self.H[self.action]
			self.H-=self.l_rate*(reward-self.r)*self.P
			self.H[self.action]=temp_h+self.l_rate*(reward-self.r)*(1-self.P[self.action])
		self.t+=1

		# Softmax
		e_H=np.exp(self.H)
		self.P=e_H/np.sum(e_H)

		# Select action based on Policy probabilities
		self.action=np.random.choice(self.n_actions,p=self.P)
		return self.action

		
