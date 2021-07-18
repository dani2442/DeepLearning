from RadialBasisFunction import *

RBF =RadialBasisFunction(3,4)

X=np.array([[1,4],[2,5],[3,6]])
Y=np.array([[1,2,3]]).T

RBF.Forward(X)
