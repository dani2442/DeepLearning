import numpy as np
from perceptron import Perceptron

P=Perceptron(5)

x=np.array([1,2,3,4,5],dtype=float)
y=np.array([-1,-2,-3,-4,-5],dtype=float)

for i in range(4):
    P.UpdateWeights(x,y)
    print(P.Loss(x,y))
