import numpy as np

class RadialBasisFunction(object):
    def __init__(self,in_size,h_size): 
        self.in_size=in_size
        self.h_size=h_size

        self.mu=np.random.randn(self.h_size,self.in_size)
        self.sigma=np.random.randn(self.h_size,1)

        self.W=np.random.randn(1,self.h_size)

    def SortestDistante(mu,x):
        d=np.sum(np.square(mu-x),axis=0)
        return np.argmin(d)

    def MaxDistante(mu):
        dis=np.zeros((len(mu),len(mu)))
        for i in range(len(mu)):
            for j in range(i):
                dis[i,j]=np.sum(np.square(mu[:,i]-mu[:,j]))
        return np.max(dis)

    def AverageDistance(mu):
        dis=np.zeros((len(mu),len(mu)))
        for i in range(len(mu)):
            for j in range(i):
                dis[i,j]=np.sum(np.square(mu[:,i]-mu[:,j]))
        return sum(dis)/(len(mu)*len(mu)-len(mu))

    def TrainHiddenLayer(self,X,iter): # Unsupervised 
        X=np.copy(X)
        self.mu=X[:,:self.h_size]

        clusters=[[] for i in range(self.h_size)]
        for it in range(iter):
            for i in range(min(150,len(X[0]))): # limit the number of samples in training hidden layer
                clusters[RadialBasisFunction.SortestDistante(self.mu,X[:,[i]])]+=[i]
            for i in range(len(clusters)):
                vec=np.zeros((self.in_size,1))
                for j in range(len(clusters[i])):
                    vec+=X[:,[clusters[i][j]]]
                vec/=len(clusters[i])
                self.mu[:,[i]]=vec
        
        self.sigma=RadialBasisFunction.MaxDistante(self.mu)/np.sqrt(self.h_size) # We could select two different sigmas
        # sigma=2*RadialBasisFunction.AverageDistance(self.mu) 
        return 

    def TrainOutputLayer(self,X,Y,batch_size,iter,lrate,lambd): # Supervised
        for it in range(iter):
            loss=0
            for i in range(len(X[0]/batch_size)):
                x=X[:,i*batch_size:(i+1)*batch_size]
                y=Y[:,i*batch_size:(i+1)*batch_size]
                output=self.Forward(x)
                self.dW=np.dot(output-y,self.H.T)
                self.W-=lrate*self.dW +lrate*lambd*self.W
                loss+=self.Loss(output,y)
            print(loss/len(X[0]))

    def Train(self,X,Y,batch_size=1,iter=100,lrate=0.01,lambd=0.01): 
        self.TrainHiddenLayer(X,iter)
        self.TrainOutputLayer(X,Y,batch_size,iter,lrate,lambd)

    def Loss(self,output,y):
        return np.sum(np.square(output-y)) # +self.lambd/2*np.sum(np.square(self.W))

    def Forward(self,x):
        self.A = np.sum(np.square(np.array([x])-np.array([self.mu]).T),axis=1)/(2*np.square(self.sigma))
        self.H = np.exp(self.A)
        self.O = np.dot(self.W,self.H)
        return self.O

