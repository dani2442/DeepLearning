from Model import Model
import numpy as np
class CarRentalModel(Model):
	def __init__(self,max_a,max_b):
		self.max_a=max_a
		self.max_b=max_b

		self.gamma=0.9

		self.V=np.zeros((max_a,max_b))
		self.P=np.zeros((max_a,max_b),dtype=int)
		self.S=[]
		for i in range(max_a):
			for j in range(max_b):
				self.S+=[(i,j)]
		self.ProbA=np.zeros((max_a,max_a))
		self.ProbB=np.zeros((max_b,max_b))

		self.requestsA=3
		self.requestsB=3
		self.returnedA=4
		self.returnedB=2

		self.InitProbabilities()

	# Matrix with the probabilities P(S'|S)
	def InitProbabilities(self):
		self.P_a_req=np.zeros((self.max_a))
		self.P_a_ret=np.zeros((self.max_a))

		self.P_b_req=np.zeros((self.max_b))
		self.P_b_ret=np.zeros((self.max_b))

		fact=1; req=1; ret=1
		for i in range(self.max_a-1):
			self.P_a_req[i]=req*np.exp(-self.requestsA)/fact
			self.P_a_ret[i]=ret*np.exp(-self.returnedA)/fact
			fact*=i+1
			req*=self.requestsA
			ret*=self.returnedA
		self.P_a_req[-1]=1-np.sum(self.P_a_req)
		self.P_a_ret[-1]=1-np.sum(self.P_a_ret)

		fact=1; req=1; ret=1
		for i in range(self.max_b-1):
			self.P_b_req[i]=req*np.exp(-self.requestsB)/fact
			self.P_b_ret[i]=ret*np.exp(-self.returnedB)/fact
			fact*=i+1
			req*=self.requestsB
			ret*=self.returnedB
		self.P_b_req[-1]=1-np.sum(self.P_b_req)
		self.P_b_ret[-1]=1-np.sum(self.P_b_ret)


	def Train(self,iter=10):
		for it in range(iter):
			self.IterativePolicyEvaluation()
			self.PolicyImprovement()

	def IterativePolicyEvaluation(self):
		V1=np.zeros((self.max_a,self.max_b))
		for it2 in range(1):
			for s in self.S:
				V1[s]=self.GetExpected(s,self.P[s])
			self.V=V1
		print(self.V)

	def GetExpected(self,s,a):
		S=(s[0]+a,s[1]-a)
		R=-2*abs(a)
		suma=0.0
		for ret_a in range(self.max_a):
			for req_a in range(self.max_a):
				s1=min(max(S[0]+ret_a-req_a,0),self.max_a-1)
				r_a= min(S[0],req_a)*10
				for ret_b in range(self.max_b):
					for req_b in range(self.max_b):
						s2=min(max(0,S[1]+ret_b-req_b),self.max_b-1)
						r_b= min(S[1],req_b)*10
						suma+=(R+r_a+r_b+self.gamma*self.V[(s1,s2)])*self.P_a_req[req_a]*self.P_a_ret[ret_a]*self.P_b_req[req_b]*self.P_b_ret[ret_b]
		return suma

	def PolicyImprovement(self):
		for s in self.S:
			a_value=np.array([self.GetExpected(s,a) for a in range(-5,5+1)])
			self.P[s]=np.argmax(a_value)-5
		print(self.P)

"""
	def GetExpected(self,s,a):
		S=(s[0]+a,s[1]-a)
		R=-2*abs(a)
		suma=0.0
		p=0
		for s_ in self.S:
			for req_a in range(self.max_a):
				ret_a=s_[0]-S[0]+req_a
				if ret_a<0 or ret_a>=self.max_a: continue
				for req_b in range(self.max_b):
					ret_b=s_[1]-S[1]+req_b
					if ret_b<0 or ret_b>=self.max_b: continue
					p+=self.P_a_req[req_a]*self.P_a_ret[ret_a]*self.P_b_req[req_b]*self.P_b_ret[ret_b]
					suma+=self.P_a_req[req_a]*self.P_a_ret[ret_a]*self.P_b_req[req_b]*self.P_b_ret[ret_b]*(R+10*(ret_a+ret_b)+self.gamma*self.V[s_])
		print(p)
		return suma
"""