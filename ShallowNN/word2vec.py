import numpy as np
import matplotlib.pyplot as plt

class word2vec(object):

    def __init__(self,rnge,lexicon,p,learning_rate=0.001):
        self.lrate = learning_rate  # Learning Rate
        self.range = rnge  # Range of words we choose as context words
        self.lexicon = lexicon
        self.p = p  # Hidden layer size
        self.u = np.random.randn(self.lexicon,p)  # Initialize weights from the first layer
        self.v = np.random.randn(p,self.lexicon)  # Initialize weights from the second layer
        
    def Loss(self,x,y):
        return -np.log(self.Output(x)[np.argmax(y)])
        
    def Forward(self,x):
        return np.dot(self.u,x).sum(axis=0)
    
    def Output(self,x):
        num = np.exp(np.dot(self.Forward(x),self.v))
        return num/np.sum(num)
    
    def Gradient(self,x,y):
        e = y - self.Output(x)
        return [-np.sum(np.dot(e,np.transpose(self.v))),-np.dot(np.transpose(e),self.forwad(x))]
    
    def UpdateWeights(self,x,y):
        self.u -= self.lrate*self.Gradient(x,y)[0]
        self.v -= self.lrate*self.Gradient(x,y)[1]
        
    def Train(self,X,Y,iter=20):
        iter_loss = np.zeros(iter)
        for it in range(iter):
            loss = 0
            
            for i in range(len(X)):
                loss += self.Loss(X[i],Y[i])
                self.UpdateWeights(X[i],Y[i])
                
            iter_loss[it] = (loss/len(X))
        return plt.plot(iter_loss)