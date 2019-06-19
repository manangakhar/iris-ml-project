import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

# shape / dimensions
print("dimensions/shape")
print(dataset.shape)

# head -  print top x
print("dataset top x")
print(dataset.head(5))
print("dataset top x")
print(dataset[:5])

# statistical summary data
print("dataset statistical data summary")
print(dataset.describe())

# group by
print("group by class")
print(dataset.groupby('class').size())
# cannot print directly, it gives the class reference instead (probably)
for key, item in dataset.groupby('class'):
    print(dataset.groupby('class').get_group(key))

# box and whisker plots - box boundary at Q1 and Q3 quartiles
# while the middle line in box is Q2 quartile (median)
dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
plt.show()

# histogram - a diagram consisting of rectangles
# whose area is proportional to the frequency of a variable
# and whose width is equal to the class interval.
dataset.hist()
plt.show()

# scatter plot matrix
scatter_matrix(dataset)
plt.show()

# split out validation dataset
array = dataset.values
X = array[:, 0:4]
# print("X") # here X has all but the last class column
# print(X) # here Y has the last class column
Y = array[:, 4]
# print("Y")
# print(Y)
validation_size = 0.2
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                random_state=seed)
# print("X_train")
# print(X_train)
# print("X_validation")
# print(X_validation)
# print("Y_train")
# print(Y_train)
# print("Y_validation")
# print(Y_validation)

seed = 7
scoring = 'accuracy'  # ratio of the number of correctly predicted instances in
# divided by the total number of instances in the dataset multiplied by 100 to give a percentage

# using the following algorithms
# Logistic Regression (LR) [linear]
# Linear Discriminant Analysis (LDA) [linear]
# K-Nearest Neighbors (KNN) [non-linear]
# Classification and Regression Trees (CART) [non-linear]
# Gaussian Naive Bayes (NB) [non-linear]
# Support Vector Machines (SVM) [non-linear]

# spot check algorithms
# model calling is possible to run in a loop as each class has implemented the function 'fit'
models = []
models.append(("LR", LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(("LDA", LinearDiscriminantAnalysis()))
models.append(("KNN", KNeighborsClassifier()))
models.append(("CART", DecisionTreeClassifier()))
models.append(("NB", GaussianNB()))
models.append(("SVM", SVC(gamma='auto')))

# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    # each element in cv_results array will have the accuracy run via 1/k of the k-cross validation
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f %f " % (name, cv_results.mean(), cv_results.std())
    print(msg)
# SVM comes out to be the best in this case

# Plot and compare algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111) # xyz : xXy grid with z subplot
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# following with KNN as it is simple
print("working on KNN")
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print("accuracy score")
print(accuracy_score(Y_validation, predictions))
print("confusion matrix")
print(confusion_matrix(Y_validation, predictions))
print("classification report")
print(classification_report(Y_validation, predictions))