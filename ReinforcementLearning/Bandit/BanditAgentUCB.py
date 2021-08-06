import numpy as np
from Agent import Agent
class BanditAgentUCB(Agent):
	def __init__(self,n,exploration=2,eps=0,l_rate=None,init=0):
		self.n_actions=n

		self.t=0
		self.N=np.zeros((n),dtype=int)
		self.Q=np.full((n),init)

		self.action=None
		self.eps=eps
		self.c=exploration
		self.l_rate=l_rate

	def Action(self,state,reward):
		if reward!=None:
			self.N[self.action]+=1
			self.t+=1

			# Stationary vs Non-Stationary
			l_rate=1/self.N[self.action] if self.l_rate is None else self.l_rate 

			self.Q[self.action]+=l_rate*(reward-self.Q[self.action])

		# eps-greedy selection
		if np.random.uniform()<self.eps:
			self.action=np.random.randint(0,self.n_actions)
		else:
			self.action=np.argmax(self.Q+self.c*np.sqrt(np.log(self.t)/(self.N+0.00001)))

		return self.action