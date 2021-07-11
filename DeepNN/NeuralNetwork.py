import numpy as np
from ParameterLearning.Stochastic import Stochastic

class NeuralNetwork(object):
    def __init__(self): 
        self.layers=[]

    def Forward(self,x): 
        for i in range(len(self.layers)):
            x=self.layers[i].Forward(x)

    def Backward(self,output,y): 
        dO=self.lossFunction.Backward(output,y)
        for i in range(len(self.layers)-1,0,-1): 
            dO=self.layers[i].Backward(dO)

    def UpdateParameters(self,learningmethod): 
        for layer in self.layers:
            layer.UpdateParameters(learningmethod)

    def Train(self,X,Y,iter=10,batch_size=2,learningmethod=Stochastic): # TODO: batch implementation
        for it in range(iter):
            loss=0
            for i in range(len(X)):
                out=self.Forward(X[:,[i]])
                loss+=self.GetLoss(out,Y[:,[i]])
                self.Backward(out,Y[:,[i]])
                self.UpdateParameters(learningmethod)
            print(loss/len(X))

    def AddLayer(self,layer): self.layers+=[layer]

    def SetLossFunction(self,lossFunction): self.lossFunction=lossFunction

    def GetLoss(self,output,y): return self.lossFunction.Loss(output,y)