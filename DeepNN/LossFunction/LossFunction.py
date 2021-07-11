class LossFunction(object):
    def Loss(self,output,y): raise Exception("Loss: Function not implemented")

    def Backward(self,x,y): raise Exception("Backward: Function not implemented")