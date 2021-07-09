import numpy as np

class Perceptron(object):
    def __main__(self,size):
        self.size=size
        self.w=np.random.randn(size) # Initialize weights using normal distribution

    def forward(self,x):
        return np.sign(np.dot(self.w,x))

    