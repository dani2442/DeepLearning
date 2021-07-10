import numpy as np

# TODO: multiple Loss functions
# TODO: share W,V param (2.5.1.3)
# TODO: Nonlinear Activation (2.5.2) 
# TODO: Deep Autoencoders, Multiple layers (2.5.3)


class Autoencoder(object):
    def __init__(self,size_in,size_out,lrate=0.01,parameter_share=False):
        self.W=np.random.randn(size_in,size_out)
        self.V=np.random.randn(size_out,size_in)

        self.lrate=lrate
        self.param_share=parameter_share 

    def Encode(self,x):
        return np.dot(self.W.T,x.T)

    def Forward(self,x):
        self.o=np.dot(self.W.T,x.T)
        self.output=np.dot(self.V.T,self.o)

    def CalculateGradient(self,x,y):
        self.dL=self.output-y.T
        self.dV=np.dot(self.o,self.dL.T)
        self.dW=np.dot(np.dot(self.V,self.dL),x).T
        dd=5

    def UpdateWeights(self):
        self.V-=self.lrate*self.dV
        self.W-=self.lrate*self.dW

    def Train(self,X,Y,iter=100):
        self.iter=iter
        for it in range(iter):
            loss=0
            for i in range(len(X)):
                self.Forward(X[[i]])
                self.CalculateGradient(X[[i]],Y[[i]])
                loss+=self.Loss(X[[i]],Y[[i]])

                self.UpdateWeights()
            if it%10==0: print(loss/iter)

    def Loss(self,x,y):
        return np.sum(np.square(self.output-y.T))/2

print("heheh2")