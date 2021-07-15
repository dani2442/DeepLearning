import numpy as np
import copy
from ActivationFunction.ActivationFunction import *
from LossFunction.LossFunction import *
from Layer.Layer import *
from ParameterLearning.ParameterLearning import *

class NeuralNetwork(object):
    def __init__(self,in_size,out_size):
        self.in_size=in_size
        self.out_size=out_size 
        self.layers=[]

    def Forward(self,x): 
        for i in range(len(self.layers)):
            x=self.layers[i].Forward(x)
        return x

    def Backward(self,x,output,y): 
        dO=self.lossFunction.Backward(output,y)
        if len(self.layers)==1:
            self.layers[0].Backward(x,dO)
        else:
            n=len(self.layers)-1
            dIn=self.layers[n].Backward(self.layers[n-1].O,dO)
            for i in range(n-1,0,-1): 
                dIn=self.layers[i].Backward(self.layers[i-1].O,dIn)
            self.layers[0].Backward(x,self.layers[1].dIn)

    def UpdateParameters(self): 
        for layer in self.layers:
            layer.UpdateParameters()

    def Train(self,X,Y,iter=100,batch_size=1,learningmethod=Stochastic,lossFunction=MSE): # TODO: batch implementation
        self.SetLossFunction(lossFunction)
        self.SetLearningMethod(learningmethod)

        for it in range(iter):
            loss=0
            for i in range(int(len(X[0])/batch_size)):
                s,f=i*batch_size,(i+1)*batch_size
                out=self.Forward(X[:,s:f])
                loss+=self.GetLoss(out,Y[:,s:f])
                self.Backward(X[:,s:f],out,Y[:,s:f])
                self.UpdateParameters()
            print(loss/len(X[0]))

    def AddLayer(self,layer): self.layers+=[layer]

    def SetLearningMethod(self,learningmethod):
        for i in self.layers:
            i.SetLearningMethod(learningmethod())

    def SetLossFunction(self,lossFunction): self.lossFunction=lossFunction

    def GetLoss(self,output,y): return self.lossFunction.Loss(output,y)