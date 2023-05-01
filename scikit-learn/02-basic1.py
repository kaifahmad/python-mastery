import matplotlib as plt
import numpy as np
from sklearn import datasets, linear_model, metrics

diabetes = datasets.load_diabetes()
diabetes_x = diabetes.data
diabetes_x_train = diabetes_x[:-30]
diabetes_y_train = diabetes.target[:-30]
diabetes_x_test = diabetes_x[-30:]
diabetes_y_test = diabetes.target[-30:]

model = linear_model.LinearRegression()
model = model.fit(diabetes_x_train,diabetes_y_train)

diabetes_y_predicted = model.predict(diabetes_x_test)
print(f"Mean Square Error: {metrics.mean_squared_error(y_true=diabetes_y_test,y_pred=diabetes_y_predicted)}")
