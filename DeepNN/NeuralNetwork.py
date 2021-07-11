import numpy as np

class NeuralNetwork(object):
    def __init__(self): 
        self.layers=[]

    def Forward(self,X): pass

    def Backward(self,output,y): pass

    def Train(self,X,Y,iter=10,batch_size=2): pass

    def AddLayer(self,layer): self.layers+=[layer]

    def SetLossFunction(self,lossFunction): self.lossFunction=lossFunction

    def GetLoss(self,output,y): return self.lossFunction.Loss(output,y) 