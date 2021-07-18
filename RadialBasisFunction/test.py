from RadialBasisFunction import *
import numpy as np
from sklearn import datasets

C=3
F=2
X,Y = datasets.make_classification(
    n_features=C,
    n_classes=F,
    n_samples=200,
    n_redundant=0,
    n_clusters_per_class=1
)

X=X.T
Y=np.array([Y])


RBF =RadialBasisFunction(C,4)
RBF.Train(X,Y,5,100)
