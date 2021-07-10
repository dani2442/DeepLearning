import numpy as np
from autoencoder import Autoencoder
from sklearn import datasets

P=Autoencoder(4,2)

X, Y = datasets.make_classification(
    n_features=4,
    n_classes=4,
    n_samples=100,
    n_redundant=0,
    n_clusters_per_class=1
)

Y_=[]
for i in range(len(X)):
    vec=np.zeros((4))
    vec[Y[i]]=1
    Y_+=[vec]

P.Train(X,np.array(Y_),20)

