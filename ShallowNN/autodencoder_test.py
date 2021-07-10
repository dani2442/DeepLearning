import numpy as np
from autoencoder import Autoencoder
from sklearn import datasets

P=Autoencoder(4,2)

X, Y = datasets.make_classification(
    n_features=4,
    n_classes=4,
    n_samples=200,
    n_redundant=0,
    n_clusters_per_class=1
)

P.Train(X,Y,2)

