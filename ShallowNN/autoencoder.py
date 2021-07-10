import numpy as np

# TODO: multiple Loss functions
# TODO: share W,V param (2.5.1.3)
# TODO: Nonlinear Activation (2.5.2) 
# TODO: Deep Autoencoders, Multiple layers (2.5.3)


class Autoencoder(object):
    def __init__(self,size_in,size_out,lrate=0.01,parameter_share=False):
        self.W=np.random.randn(size_in,size_out)
        self.V=np.random.randn(size_out,size_in)

        self.lrate=0.01
        self.param_share=parameter_share 

    def Forward(self,x):
        return np.dot(self.V.T,np.dot(self.W.T,x))

    def CalculateGradient(self,output,y):
        self.dV=self.V*self.GradientLoss(output,y)
        self.dW=np.dot(self.W,self.dV)

    def UpdateWeights(self):
        self.V-=self.lrate*self.dV
        self.W-=self.lrate*self.dW

    def Train(self,X,Y,iter=100):
        self.iter=iter
        for it in range(iter):
            loss=0
            for i in range(len(X)):
                self.CalculateGradient(X[i],Y[i])
                loss+=self.Loss(X[i],Y[i])

                self.UpdateWeights()

            print(loss/iter)

    def Loss(self,output,y):
        return np.sum(np.square(output-y))/2

    def GradientLoss(self,output,y):
        return output-y