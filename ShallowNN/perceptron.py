import numpy as np

class Perceptron(object):
    def __init__(self,size,learning_rate=0.001,regularization_param=0.1):
        self.gradient=np.zeros(size)
        self.loss=None
        self.rrate=learning_rate  # Regularization constant
        self.lrate=regularization_param  # Learning rate
        self.size=size
        self.w=np.random.randn(size) # Initialize weights using normal distribution

    def Forward(self,x):
        return np.sign(np.dot(self.w,x))

    def CalculateGradient(self,x,y):
        self.gradient+=(y-self.Forward(x))*x  # Update gradient using batch descent

    def UpdateWeights(self): 
        self.w=self.w*(1-self.rrate*self.lrate)+self.lrate*self.gradient/self.iter

    def Train(self,X,Y,iter=100):
        self.iter=iter
        for it in range(iter):
            loss=0 # Meassure the error

            for i in range(len(X)): # Calculate the gradient and loss of the batch
                self.CalculateGradient(X[i],Y[i])
                loss+=self.Loss(X[i],Y[i])

            self.UpdateWeights()
            print(loss/iter)

    def Loss(self,x,y):
        return np.sum(np.square(y-self.Forward(x)))