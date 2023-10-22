import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
iris_data = pd.read_csv(r'/Users/moinmairaj/Desktop/Data Science/IRIS.csv')
print(iris_data.head())
print(iris_data.info())
print(iris_data.describe(include='all'))
print(iris_data.isnull().values.any())  # no null value
print(iris_data.duplicated().any())
iris_data = iris_data.drop_duplicates()
print(iris_data.duplicated().any())
print(iris_data.info())

# visulaize data
print(iris_data["species"].value_counts())
sns.set()
sns.countplot(x="species", data=iris_data)
plt.show()
sns.countplot(x="petal_width", data=iris_data)
plt.show()
sns.countplot(x="petal_length", hue='species', data=iris_data)
plt.show()
sns.countplot(x="sepal_width", hue='species', data=iris_data)
plt.show()
sns.countplot(x="sepal_length", hue='species', data=iris_data)
plt.show()
sns.barplot(x='species', y='petal_length', data=iris_data)
plt.title("Species vs petal length")
plt.show()
# petal length order setosa < vesicolor < virginica mean values
sns.barplot(x='species', y='petal_width', data=iris_data)
plt.title("Species vs petal width")
plt.show()
# petal width order setosa < vesicolor < virginica mean values
sns.barplot(x='species', y='sepal_width', data=iris_data)
plt.title("Species vs sepal width")
plt.show()
# petal width order setosa > virginica >=  vesicolor mean values
sns.barplot(x='species', y='sepal_length', data=iris_data)
plt.title("Species vs sepal length")
plt.show()
# sepal length order setosa < vesicolor < virginica mean values
# processing species from categorical to numeric
# setosa = 0, virginica = 1, versicolor = 2
iris_data.replace({'species':{'Iris-setosa':0, 'Iris-virginica':1, 'Iris-versicolor':2}}, inplace=True)
print(iris_data.head())
print(iris_data.info())
# dropping colum species and scoring that in Y and rest in X
X = iris_data.drop(columns=['species'], axis=1)
Y = iris_data['species']
# scaling X
scaler = StandardScaler()
X = scaler.fit_transform(X)
print(X)
print(Y)
# splitting the data  into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)
print(Y.shape, Y_train.shape, Y_test.shape)
# modelling since it is a regression problem so we use LogisticRegression
model = LogisticRegression()
model.fit(X_train, Y_train)
# Evaluating the model
X_train_prediction = model.predict(X_train)
X_test_prediction = model.predict(X_test)
# check accuracy
training_accuracy = accuracy_score(Y_train, X_train_prediction)
testing_accuracy = accuracy_score(Y_test, X_test_prediction)
print("Training data accuracy:", training_accuracy)
print("Test data accuracy:", testing_accuracy)
print("\nmodel complete as accuracy is greater than 0.75 for both test as well as training data")