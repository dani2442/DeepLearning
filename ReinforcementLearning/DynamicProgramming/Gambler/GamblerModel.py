from Model import Model
import numpy as np
class GamblerModel(Model):
    def __init__(self,p):
        self.prob=p
        self.gamma=0.9
        self.V=np.zeros((100))
        self.P=np.zeros((100))


    def Train(self,iter=10):
        # Update value function
        for it in range(iter):
            V1=np.zeros((100))
            for s in range(100):
                max_value=0
                for a in range(1,s+1):
                    new_v=self.GetExpected(s,a)
                    if new_v>max_value: 
                        max_value=new_v
                V1[s]=max_value
            self.V=V1
        
        # Update policy function
        for s in range(100):
            max_value=0;max_id=0
            for a in range(1,s+1):
                new_v=self.GetExpected(s,a)
                if new_v>max_value:
                    max_value=new_v
                    max_id=a
            self.P[s]=max_id
        
        print(self.V)
        print(self.P)

    def GetExpected(self,s,a):
        v1,r=(0,1) if s+a>=100 else (self.V[s+a],0)
        v2=0 if s-a<=0 else self.V[s-a]
        return self.prob*(r+self.gamma*v1)+(1-self.prob)*v2
        