import numpy as np
from perceptron import Perceptron
import sklearn
from sklearn import datasets

P=Perceptron(2)

X, Y = datasets.make_classification(
    n_features=2,
    n_classes=2,
    n_samples=200,
    n_redundant=0,
    n_clusters_per_class=1
)

P.Train(X,Y,100)

