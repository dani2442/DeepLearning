from os import stat
import numpy as np
import codecs,json,copy
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

    def Train(self,X,Y,iter=100,batch_size=1,learningmethod=Stochastic(),lossFunction=MSE): # TODO: batch implementation
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
            if it%10==0:print(loss/len(X[0]))

    def AddLayer(self,layer): self.layers+=[layer]

    def SetLearningMethod(self,learningmethod):
        for i in self.layers:
            i.SetLearningMethod(copy.deepcopy(learningmethod))

    def SetLossFunction(self,lossFunction): self.lossFunction=lossFunction

    def GetLoss(self,output,y): return self.lossFunction.Loss(output,y)

    def Export(self,path): 
        layers=[]
        for i in self.layers:
            layer=dict()

            if i.ActivationFunction==Sigmoid: layer["ActivationFunction"]="Sigmoid"
            elif i.ActivationFunction==Tanh: layer["ActivationFunction"]="Tanh"
            elif i.ActivationFunction==ReLU: layer["ActivationFunction"]="ReLU"

            if type(i)==Plain:
                layer["TypeLayer"]="Plain"
                layer["W"]=i.W.tolist()
                layer["B"]=i.B.tolist()

            layers+=[layer]
        json.dump(layers, codecs.open(path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

    @staticmethod
    def Export2(nn,path):
        nn.Export(path)

    def Import(self,path): 
        obj_text = codecs.open(path, 'r', encoding='utf-8').read()
        obj=json.loads(obj_text)
        self.layers=[]
        for i in obj:
            if i["TypeLayer"]=="Plain":
                W=np.array(i["W"])
                B=np.array(i["B"])
                l=Plain(len(W[0]),len(W))
                l.W=W
                l.B=B
            
            if i["ActivationFunction"]=="Sigmoid": l.ActivationFunction=Sigmoid
            elif i["ActivationFunction"]=="Tanh": l.ActivationFunction=Tanh
            elif i["ActivationFunction"]=="ReLU": l.ActivationFunction=ReLU

            self.layers+=[l]

        self.in_size=self.layers[0].in_size
        self.out_size=self.layers[-1].out_size

    @staticmethod
    def Import2(path):
        nn=NeuralNetwork(0,0)
        nn.Import(path)
        return nn
            


            