import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression(object):

    def __init__(self,size,learning_rate=0.001,regularization_param=0.1):
        self.rrate = regularization_param  # Regularization Constant
        self.lrate = learning_rate  # Learning Rate
        self.size = size
        self.w = np.random.randn(size)  # Initialize weights using normal distribution
        
    def forward(self,x):
        return 1/(1+np.e**(-np.dot(self.w,x)))
    
    def loss(self,x,y):
        return -np.log(np.abs(y/2 - 0.5 + self.forward(x)))
    
    def gradient(self,x,y):
        return -(y*x)*self.forward(-y*x)
    
    def updateweights(self,x,y):
        self.w = self.w*(1-self.lrate*self.rrate) - self.lrate*self.gradient(x,y)
        
    def train(self,X,Y,iter=200):
        iter_loss=np.zeros(iter)
        for it in range(iter):
            loss=0
            
            for i in range(len(X)):
                loss+=self.loss(X[i],Y[i])
                self.updateweights(X[i],Y[i])
                
            iter_loss[it]=(loss/len(X))
        return plt.plot(iter_loss)
