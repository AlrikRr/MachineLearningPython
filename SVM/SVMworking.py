import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

#print(cancer.feature_names) # check features on the dataset
#print(cancer.target_names) # check names on the dataset

# Init x and y
x = cancer.data # cancer.data[:100]
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

#print(x_train, y_train)
classes = ['malignant', 'benign'] # 0 1 

# First Try without parameters : Acc is bad
#clf = svm.SVC()
# Second Try with kernel parameters
    # linear : much better accu / C=2 is the soft margin
    # poly : much better but damn it takes a long time to train ! / degree=2 for less time i guess
clf = svm.SVC(kernel="linear", C=2)
clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)

print("SVM Acc:" , acc)

# Test with the KNN
clf2 = KNeighborsClassifier(n_neighbors=13) # Increese the neighbors for better acc
clf2.fit(x_train,y_train)
y_pred2 = clf2.predict(x_test)
acc2 = metrics.accuracy_score(y_test, y_pred2)
print("KNN Acc:", acc2)
