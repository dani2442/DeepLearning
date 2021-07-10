
from logistic_regression import LogisticRegression
from sklearn import datasets

L=LogisticRegression(2)

X,Y = datasets.make_classification(
    n_features=2,
    n_classes=2,
    n_samples=200,
    n_redundant=0,
    n_clusters_per_class=1
)

L.train(X,Y)