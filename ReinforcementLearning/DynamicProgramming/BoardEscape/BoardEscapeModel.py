from Model import Model
import numpy as np
class BoardEscapeModel(Model):
	def __init__(self,n,ProbActions=[0.25,0.25,0.25,0.25],stop_states=[[0,0],[3,3]],reward=np.full((4,4),-1)):
		self.n=n
		self.P=ProbActions
		self.Stop=[n*s[0]+s[1] for s in stop_states]
		self.R=reward
		self.action=np.array([[1,0],[-1,0],[0,1],[0,-1]])
		self.V=np.zeros((n,n))
		self.S=[]
		for i in range(n):
			for j in range(n):
				self.S+=[(i,j)]

	def Train(self,iter=10):
		for it in range(iter):
			self.V=self.IterativePolicyEvaluation()
		
		print(self.V)
			
	# Iterative Policy Evaluation
	def IterativePolicyEvaluation(self):
		V1=np.zeros((self.n,self.n))
		for i in range(len(self.S)):
			if i in self.Stop: continue
			for a in range(len(self.P)):
				new_s=self.S[i]+self.action[a]
				if not np.prod((new_s>=0)*(new_s<self.n)):
					 new_s=self.S[i]
				new_s=tuple(new_s)
				V1[self.S[i]]+=self.P[a]*(self.R[new_s]+self.V[new_s])

		return V1

	def GetArgmaxPolicy(self):
		P=np.zeros((self.n,self.n))
		for i in range(len(self.S)):
			if i in self.Stop: continue
			max_value,max_acc=-999999,0
			for a in range(len(self.P)):
				new_s=self.S[i]+self.action[a]
				if not np.prod((new_s>=0)*(new_s<self.n)):
					 new_s=self.S[i]
				new_s=tuple(new_s)
				if self.V[new_s]>max_value:
					max_value=self.V[new_s]
					max_acc=a
			P[self.S[i]]=max_acc
		return P