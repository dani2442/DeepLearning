import numpy as np
import matplotlib.pyplot as plt

class SupporVectorMachines(object):

    def __init__(self,size,learning_rate=0.001,regularization_param=0.1):
        self.rrate=regularization_param  # Regularization Constant
        self.lrate=learning_rate  # Learning Rate
        self.size=size
        self.w=np.random.randn(size)  # Initialize weights using normal distribution
        
    def loss(self,x,y):
        return max(0,1-y*self.forward(x))
        
    def forward(self,x):
        return np.dot(self.w,x)
    
    def gradient(self,x,y):
        if y*self.forward(x)<1: return -y*x
        else: return 0
    
    def upweights(self,x,y):
        self.w = self.w*(1-self.rrate*self.lrate) - self.lrate*self.gradient(x,y)
        
    def train(self,X,Y,iter=100):
        iter_loss=np.zeros(iter)
        for it in range(iter):
            loss=0
            
            for i in range(len(X)):
                loss+=self.loss(X[i],Y[i])
                self.upweights(X[i],Y[i])
                
            iter_loss[it]=(loss/len(X))
        return plt.plot(iter_loss)