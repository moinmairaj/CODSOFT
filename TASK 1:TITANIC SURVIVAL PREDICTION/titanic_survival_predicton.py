# importing the libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
# data collection and processing
titanic_data = pd.read_csv(r'/Users/moinmairaj/Desktop/Data Science/tested.csv')

# viewing data
print(titanic_data.head())
print(titanic_data.shape)
print(titanic_data.info())
# there are many null values in Cabin and age columns
print(titanic_data.isnull().sum())
# Handling the null values
# since Cabin has maximum null values also Cabin is irrelevant to survival
# so dropping Cabin column
titanic_data = titanic_data.drop(columns='Cabin', axis=1)
print(titanic_data.head())
# age is relevant for survival so can't drop that thus replacing null age values with mean age
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)
# for missing fare lets use modal value of fare
titanic_data['Fare'].fillna(titanic_data['Fare'].mode()[0], inplace=True)
# No null value in data now
print(titanic_data.isnull().sum())

# Analysis of data
print(titanic_data.describe())
print(titanic_data.info())
# count of survived and not survived
print(titanic_data['Survived'].value_counts())
print(titanic_data['Sex'].value_counts())
# out of 418, 152 survived and 256 did not.

# Data visualization
# theme setting of plots
sns.set()
# count plot for survived
sns.countplot(x='Survived', data=titanic_data)
plt.show()
# count plot for gender
sns.countplot(x='Sex', data=titanic_data)
plt.show()
# compare survived based on gender
sns.countplot(x='Sex', hue='Survived', data=titanic_data)
plt.show()
# as per the given data set all females have survived and no male has survived
# so there is some issue with dataset, anyway we will continue our model
sns.countplot(x='Pclass', hue='Survived', data=titanic_data)
plt.show()
# plot indicates chances of survival are more for class 1


# Training Part
# since Sex is text, so put male = 0 and female as 1 and also Embarked S = 0, C = 1, Q = 2
titanic_data.replace({'Sex': {'male': 0, 'female': 1}, 'Embarked': {'S': 0, 'C': 1, 'Q': 2}}, inplace=True)
print(titanic_data.head())
# separating features and target or survival
X = titanic_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Survived'], axis=1)
Y = titanic_data['Survived']
# increasing maximum iteration
scaler = StandardScaler()
# scaling X
X = scaler.fit_transform(X)
print(X)
print(Y)
# Using Logistic Regression model
# since only two output values possible
model = LogisticRegression()
# Splitting data into test and train
# use 4 arrays and split test train as 0.2 and 0.8
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

# training model
model.fit(X_train, Y_train)
# Evaluating model
X_train_prediction = model.predict(X_train)
print(X_train_prediction)
# Check accuracy of training prediction
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print("Accuracy of training data:", training_data_accuracy)
# accuracy of training date is 100%
# Check accuracy of testing prediction
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print("Accuracy of test data:", test_data_accuracy)
# accuracy of test data also comes to be 100%
# so for given data set our model is 100% accurate
