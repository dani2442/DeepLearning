import numpy as np

class RadialBasisFunction(object):
    def __init__(self,in_size,h_size): 
        self.in_size=in_size
        self.h_size=h_size

        self.MAX_it=100

        #self.mu=np.random.randn(self.h_size,1)
        self.mu=np.array([[1,4,7,10],[2,5,8,11],[3,6,9,12]])
        #self.sigma=np.random.randn(self.h_size,1)
        self.sigma=np.array([[1,2,3,4]]).T

        self.W=np.random.randn(1,self.h_size)

    def SortestDistante(mu,x):
        d=np.sum(np.square(mu-x),axis=0)
        return np.argmin(d)

    def MaxDistante(mu):
        pass

    def AverageDistance(mu):
        pass

    def TrainHiddenLayer(self,X): # Unsupervised 
        self.mu=X[:,:self.h_size]

        clusters=[[] for i in range(self.h_size)]
        for it in range(self.MAX_it):
            for i in range(len(X[0])):
                clusters[RadialBasisFunction.SortestDistante(self.mu,X[:,[i]])]+=[i]
            for i in range(len(clusters)):
                vec=np.zeros((self.in_size,1))
                for j in range(len(clusters[i])):
                    vec+=X[:,clusters[i][j]]
                vec/=len(clusters[i])
                self.mu[:,[i]]=vec
        
        sigma=RadialBasisFunction.MaxDistante(self.mu)/np.sqrt(self.h_size)
        #sigma=2*RadialBasisFunction.AverageDistance(self.mu)


    def TrainOutputLayer(self,X,Y): # Supervised
        pass

    def Train(self,X,Y): 
        self.TrainHiddenLayer(X)
        self.TrainOutputLayer(X,Y)

    def Forward(self,x):
        a=np.array([x])-np.array([self.mu]).T
        print(a)
        b= np.sum(np.square(a),axis=1)
        print(b)
        self.A =b/(2*np.square(self.sigma))
        # self.A =np.sum(np.square(np.array([x])-np.array([self.mu.T]).T),axis=1)/(2*np.square(self.sigma))
        print(self.A)
        self.H = np.exp(self.A)
        self.O = np.dot(self.W,self.H)
        return self.O

