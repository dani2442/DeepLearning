
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

def remplace(a):
    for i in range(len(a)):
        if a[i]==0:
            a[i]=-1
    return a

L.train(X,remplace(Y))