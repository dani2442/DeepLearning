from RadialBasisFunction import *
import numpy as np
from sklearn import datasets

C=3
F=1
X,Y = datasets.make_classification(
    n_features=C,
    n_classes=F,
    n_samples=200,
    n_redundant=0,
    n_clusters_per_class=1
)

def ConvertData(Y):
    Y_=[]
    for i in range(len(Y)):
        A=np.zeros((F))
        A[Y[i]]=1.0
        Y_+=[A]
    return np.array(Y_).T

X=X.T
Y=ConvertData(Y)


RBF =RadialBasisFunction(C,4)
RBF.Train(X,Y)
