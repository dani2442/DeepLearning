import numpy as np

class HopfieldNetwork:
    def __init__(self,units): 
        self.units=units
        self.b=np.random.randn(units,1)
        self.w=np.random.randn(units,units)
        for i in range(units):
            for j in range(units):
                if i<j: self.w[i,j]=self.w[j,i]

        for i in range(units): self.w[i,i]=0

    def Energy(self,s):
        return -np.dot(self.b.T,s)-np.sum(self.w*np.dot(s,s.T))/2

    def UpdateState(self,s):
        self.dE=self.b+np.dot(self.w,s)
        s[self.dE>=0]=1
        s[self.dE<0]=0

    def Train(self,X,batch_size=4,iter=10):
        X1=X-0.5
        
        for i in range(iter):
            for it in range(X.shape[1]//batch_size):
                Xb=X1[it*batch_size:(it+1)*batch_size,:]

                B=Xb.reshape((Xb.shape[0],self.units,1))
                C=Xb.reshape((Xb.shape[0],1,self.units))

                self.w+=4*np.sum(B*C,axis=0)
                self.b+=2*np.sum(Xb,axis=0).reshape((self.units,1))
        self.w/=iter
        self.b/=iter

    def Predict(self,s=None,iter=100):
        if s==None: s=np.random.randn(self.units,1)
        for i in range(self.units):
            self.w[i,i]=0
        for it in range(iter):
            self.UpdateState(s)
            if it%25==0: print(self.Energy(s))
        return s
