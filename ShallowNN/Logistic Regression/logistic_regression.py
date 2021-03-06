import numpy as np
import matplotlib.pyplot as plt

# 2.2.3 
class LogisticRegression(object):

    def __init__(self,size,learning_rate=0.001,regularization_param=0.1):
        self.rrate = regularization_param  # Regularization Constant
        self.lrate = learning_rate  # Learning Rate
        self.size = size
        self.w = np.random.randn(size)  # Initialize weights using normal distribution
        
    def forward(self,x):
        return 1/(1+np.exp(-np.dot(self.w,x)))
    
    def loss(self,x,y):
        return -np.log(np.abs(y/2 - 0.5 + self.forward(x)))
    
    def gradient(self,x,y):
        return -y*x/(1+np.exp(y*np.dot(self.w,x)))
        #return (-(y/2 - 0.5 + self.forward(x))/(y/2 - 0.5 + self.forward(x))**2)*(x*np.exp(-np.dot(self.w,x)))*(self.forward(x)**2)
    
    def updateweights(self,x,y):
        self.w = self.w*(1-self.lrate*self.rrate) - self.lrate*self.gradient(x,y)
        
    def train(self,X,Y,iter=500):
        iter_loss=[]
        for it in range(iter):
            loss=0
            
            for i in range(len(X)):
                loss+=self.loss(X[i],Y[i])
                self.updateweights(X[i],Y[i])
            
            print(loss)
            #iter_loss+=[loss/len(X)]
        #return plt.plot(iter_loss)