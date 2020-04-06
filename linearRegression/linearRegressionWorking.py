import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")  # Load student-mat.csv with pandas / sep = separators
# G1 sem1 final grade / G2 sem2 final grade / G3 exam final grade
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]  # Get the data that we want

# Now we want to predict G3 = Final Grade
predict = "G3"

# Training data
x = np.array(data.drop([predict], 1))  # New datafram witouht predict content
y = np.array(data[predict])  # Get only predic content

# We need to copy this part here because we still using x_train and y_train below
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# P2= Train the modal utils we have a better score
# P1 =We skip the training data because it's saved on the pickle file
"""
best = 0
for _ in range(100):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
   
    # x_train gonna be a section of x[] , y_train too
    # x_test/y_test used to test accuracy of the model that we are gonna create
    # test_size = split 10% of data into x_test/y_test  = Jeu d'essai

    # Training data
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)  # fit the data to find the best line
    acc = linear.score(x_test, y_test)  # return accuracy of our modal
    print(acc)
    if acc > best:
        best = acc
        # Save pickle file on directory to use it later
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)"""

# Read pickle
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

# Y = Mx + b
print('Coefficient: \n', linear.coef_)  # Coef are actually M
print('Intercept: \n', linear.intercept_)

# Nice but now we need to predict G3 for each students
predictions = linear.predict(x_test)

for i in range(len(predictions)):
    print(predictions[i], x_test[i], y_test[i])  # Predictions [ G1 G2 studytime failures absences ] G3

# graphic part
p = "G1"
style.use("ggplot")
# set graph with x and y
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")  # G3
pyplot.show()
