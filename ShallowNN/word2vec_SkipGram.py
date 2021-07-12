import numpy as np
import matplotlib.pyplot as plt

class word2vec_SkipGram(object):
    
    # Each sample contains a (1xd) row vector x, where d is the lexicon and the only non-zero possition is the one 
    #   for which our word holds place, and a (mxd) row vector with the context words and theirs positions given x.

    def __init__(self,rnge,lexicon,p,learning_rate=0.001):
        self.lrate = learning_rate  # Learning Rate
        self.range = rnge  # Range of words we choose as context words
        self.lexicon = lexicon
        self.p = p  # Hidden layer size
        self.u = np.random.randn(self.lexicon,p)  # Initialize weights from the first layer
        self.v = np.random.randn(p,self.lexicon)  # Initialize weights from the second layer
        
    def Loss(self,x,y):
        return -np.sum(np.dot(y,np.transpose(self.Output(x))))
        
    def Forward(self,x):
        return np.dot(x,self.u)
    
    def Output(self,x):
        num = np.exp(np.dot(self.Forward(x),self.v))
        return num/np.sum(num)
    
    def Gradient(self,x,y):
        e = y - np.tile(self.Output(x),(self.rnge,1))
        sum_e = np.sum(e,axis=0)
        return [-np.dot(sum_e,np.transpose(self.v)),-np.dot(np.transpose(self.Forward(x)),sum_e)]
    
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
