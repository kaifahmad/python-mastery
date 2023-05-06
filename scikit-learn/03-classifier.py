from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier 

iris = datasets.load_iris()

all_keys = iris.keys()
features = iris.data[:-1]
target_labels = iris.target[:-1]
print(all_keys)

# Training the Classifier
k_near_classifier = KNeighborsClassifier()
k_near_classifier.fit(features,target_labels)

# Cross-checking prediction with Actual label
print(f"{k_near_classifier.predict(features[-1:])}, Actual : {target_labels[-1:]}")
