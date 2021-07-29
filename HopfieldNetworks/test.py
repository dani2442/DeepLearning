from HopfieldNetwork import HopfieldNetwork
import numpy as np


X=[[1,0,1,0,0,0],[1,1,0,0,0,0],[0,0,0,1,0,1],[0,0,0,0,1,1]]
X=np.array(X)

Hop = HopfieldNetwork(6)
Hop.Train(X,batch_size=1,iter=1000)


print(Hop.Predict())