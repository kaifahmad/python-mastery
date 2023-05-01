from sklearn import datasets
iris = datasets.load_iris()

features = iris.keys()
_DESC = iris.DESCR
print(_DESC)