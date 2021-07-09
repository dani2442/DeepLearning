import numpy as np

class Perceptron(object):
    def __init__(self,size,learning_rate=0.9,regularization_param=0.01):
        self.loss=None
        self.rrate=learning_rate  # Regularization constant
        self.lrate=regularization_param  # Learning rate
        self.size=size
        self.w=np.random.randn(size) # Initialize weights using normal distribution

    def Forward(self,x):
        return np.sign(np.dot(self.w,x))

    def UpdateWeights(self,x,y):
        print(self.lrate*(y-self.Forward(x))*x)
        self.w=self.w*(1-self.rrate*self.lrate)-self.lrate*(y-self.Forward(x))*x

    def Loss(self,x,y):
        return np.sum(np.abs(y-self.Forward(x)))