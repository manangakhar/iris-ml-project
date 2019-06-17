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

