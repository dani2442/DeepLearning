import numpy as np

class RadialBasisFunction(object):
    def __init__(self,in_size,h_size,out_size): 
        self.in_size=in_size
        self.out_size=out_size
        self.h_size=h_size

        self.mu=np.random.randn(self.h_size,1)
        self.sigma=np.random.randn(self.h_size,1)

        self.W=np.random.randn(1,self.h_size)


    def Forward(x):
        self.A=x
        # dddl
