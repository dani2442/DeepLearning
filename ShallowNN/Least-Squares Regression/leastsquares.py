import numpy as np
import matplotlib.pyplot as plt

class LeastSquaresRegression(object):

    def __init__(self,size,learning_rate=0.001,regularization_param=0.1):
        self.rrate=regularization_param  # Regularization Constant
        self.lrate=learning_rate  # Learning Rate
        self.size=size
        self.w=np.random.randn(size)  # Initialize weights using normal distribution
        
    def Loss(self,x,y):
        return (y-np.dot(self.w,x))**2
        
    def Forward(self,x):
        return np.dot(self.w,x)
    
    def CalculateGradient(self,x,y):
        return -2*(y-np.dot(self.w,x))*x
    
    def UpdateWeights(self,x,y):
        self.w = self.w*(1-self.rrate*self.lrate) + 2*self.lrate*(y-np.dot(self.w,x))*x
        
    def Train(self,X,Y,iter=20):
        iter_loss=np.zeros(iter)
        for it in range(iter):
            loss=0
            
            for i in range(len(X)):
                loss+=self.Loss(X[i],Y[i])
                self.UpdateWeights(X[i],Y[i])
                
            iter_loss[it]=(loss/len(X))
        return plt.plot(iter_loss)
