import numpy as np

# TODO: multiple Loss functions
# TODO: share W,V param (2.5.1.3)
# TODO: Nonlinear Activation (2.5.2) 
# TODO: Deep Autoencoders, Multiple layers (2.5.3)


class Autoencoder(object):
    def __init__(self,size_in,size_out,parameter_share=False):
        self.W=np.random.randn((size_in,size_out))
        self.V=np.random.randn((size_out,size_in))

        self.param_share=parameter_share 

    def Forward(self,x):
        return np.dot(self.V.T,np.dot(self.W.T,x))

    def CalculateGradient(self,x,y):
        pass

    def UpdateWeights(self):
        pass

    def Train(self,X,Y):
        pass

    def Loss(self,output,y):
        return np.sum(np.square(output-y))/2