import numpy as np

class NeuralNetwork(object):
    def __init__(self): 
        self.layers=[]

    def Forward(self,x): 
        for i in range(len(self.layers)):
            x=self.layers[i].Forward(x)

    def Backward(self,output,y): pass

    def UpdateParameters(self): 
        for layer in self.layers:
            layer.UpdateParameters()

    def Train(self,X,Y,iter=10,batch_size=2): # TODO: batch implementation
        for it in range(iter):
            loss=0
            for i in range(len(X)):
                out=self.Forward(X[:,[i]])
                loss+=self.GetLoss(out,Y[:,[i]])
                self.Backward(out,Y[:,[i]])
                self.UpdateParameters()
            print(loss/len(X))

    def AddLayer(self,layer): self.layers+=[layer]

    def SetLossFunction(self,lossFunction): self.lossFunction=lossFunction

    def GetLoss(self,output,y): return self.lossFunction.Loss(output,y)