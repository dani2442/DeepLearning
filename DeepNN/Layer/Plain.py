from Layer import Layer

class Plain(Layer):
    def __init__(self): super().__init__()

    def Forward(self,x): pass
    def Backward(self): pass