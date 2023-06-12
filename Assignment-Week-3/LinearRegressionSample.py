import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import  mean_squared_error

diabetes = datasets.load_diabetes()  # this loads the data set

diabetes_X = diabetes.data  # storing the features of the data set.

diabetes_X_train = diabetes_X[:-30]  # the x data points using which our model is going to be trained
diabetes_X_test = diabetes_X[-30:]  # the x data points using which we will test our model

diabetes_y_train = diabetes.target[:-30] #  the target data.
diabetes_y_test = diabetes.target[-30:]  # the target data

model = linear_model.LinearRegression()  #using linearregression from linear_model.

model.fit(diabetes_X_train, diabetes_y_train)  # we want to fit the training data in our model.

diabetes_y_predicted = model.predict(diabetes_X_test)  # predicting the "future data" based on the test data.

print("Mean squared error is: ", mean_squared_error(diabetes_y_test, diabetes_y_predicted))  # sum of errors squared.

print("Weights: ", model.coef_) # returns the slope
print("Intercept: ", model.intercept_) # returns the intercept

# if we want to plot the model based on a single feature, since we can't plot multidimensional stuff.)
# plt.scatter(diabetes_X_test, diabetes_y_test)
# plt.plot(diabetes_X_test, diabetes_y_predicted)
#
# plt.show()

# for a single feature of index 2 from the dataset, if we will check the slope and intercept it will come out to be this.
# Mean squared error is:  3035.0601152912695
# Weights:  [941.43097333]
# Intercept:  153.39713623331698