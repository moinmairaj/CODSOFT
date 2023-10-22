import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
car_data = pd.read_csv("/Users/moinmairaj/Desktop/Data Science/car_purchasing.csv", encoding='latin1')
print(car_data.head())
print(car_data.shape)
print(car_data.info())
print(car_data.isnull().values.any())
# our data set has no null values
# impact of various features on target variable can be accessed using scatter plot
print(car_data.columns)
features = ['customer name', 'customer e-mail', 'country', 'gender', 'age', 'annual Salary',
            'credit card debt', 'net worth']
target = 'car purchase amount'
sns.countplot(x='car purchase amount', data=car_data)
plt.show()
for feature in features:
    plt.figure(figsize=(8, 6))
    plt.scatter(x=car_data[feature], y=car_data[target], alpha=0.5)
    plt.title(f'Scatter Plot of {feature} vs {target}')
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.show()

plt.figure(figsize=(8, 6))
car_data.boxplot(column='car purchase amount', by='customer name', rot=90)
plt.title('Effect of Name on Car Purchase Amount')
plt.ylabel('Car Purchase Amount')
plt.xlabel('Name')
plt.show()

plt.figure(figsize=(8, 6))
car_data.boxplot(column='car purchase amount', by='customer e-mail', rot=90)
plt.title('Effect of Email on Car Purchase Amount')
plt.ylabel('Car Purchase Amount')
plt.xlabel('Email')
plt.show()

plt.figure(figsize=(8, 6))
car_data.boxplot(column='car purchase amount', by='country', rot=90)
plt.title("Effect of country on Purchase amount")
plt.xlabel('Country')
plt.ylabel("Car Purchase Amount")
plt.show()

plt.figure(figsize=(8, 6))
car_data.boxplot(column='car purchase amount', by='net worth', rot=90)
plt.title("Effect of net worth on Purchase amount")
plt.xlabel('net worth')
plt.ylabel("Car Purchase Amount")
plt.show()
''' from the various data visualizing techniques qe inferred that:
1. Scatter plots indicate customer name or email does not have much impact on target(car purchasing amount) variable
2. All other features are important. So we will be dropping name and email columns.
3. Country does not have much affect but we will still use that.
4. Since country being a categorical data needs to be transformed into numerical data.
5. so we will use encoding techniques- we will us Target or Mean Encoding 
6. In target encoding we replace categorical data with the mean of the target variable,
for the rows of that categorical data.'''
print(car_data.duplicated().any())
# no duplicate data
# data preprocessing
car_data = car_data.drop(columns=['customer name', 'customer e-mail'], axis=1)
print(car_data.head())
print(car_data.info())
# handling country column
print(car_data.groupby('country')['car purchase amount'].mean().sort_values())
# so there are 211 unique countries and also countries car amount rated do vary,
# so we use target encoding
car_data.replace({'country': car_data.groupby('country')['car purchase amount'].mean()}, inplace=True)
print(car_data.head())
print(car_data.info())
# separating features and target as X and Y
X = car_data.drop(columns=['car purchase amount'], axis=1)
Y = car_data['car purchase amount']
# scaling X
scalar = StandardScaler()
X = scalar.fit_transform(X)
print(X)
print(Y)
# test train split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)
# model selection and training
model = LinearRegression()
model.fit(X_train, Y_train)
# Predicting and Evaluating the model
X_train_prediction = model.predict(X_train)
X_test_prediction = model.predict(X_test)
# Evaluating using mse, mae, rmse, r2 metrics
mse_train = mean_squared_error(Y_train, X_train_prediction)
rmse_train = np.sqrt(mse_train)
mae_train = mean_absolute_error(Y_train, X_train_prediction)
r2_train = r2_score(Y_train, X_train_prediction)
mse_test = mean_squared_error(Y_test, X_test_prediction)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(Y_test, X_test_prediction)
r2_test = r2_score(Y_test, X_test_prediction)
print("EVALUATION OF THE MODEL")
print("Metric:       ", " Train", " "*16, "Test")
print("1.MSE:   ", mse_train, " "*3, mse_test)
print("2.RMSE:  ", rmse_train, " "*3, rmse_test)
print("3.MAE:   ", mae_train, " "*3, mae_test)
print("4.R2:    ", r2_train, " "*3, r2_test)
'''Based on the values obtained, the model seems to perform exceptionally well,
as indicated by the high R2 value close to 1 and the relatively low values for MSE, RMSE, and MAE'''
# residual analysis
# training data
residuals_train = Y_train - X_train_prediction
plt.figure(figsize=(8, 6))
sns.residplot(x=X_train_prediction, y=residuals_train)
plt.title("Residual analysis of training data")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.show()
# test data
residuals_test = Y_test - X_test_prediction
plt.figure(figsize=(8, 6))
sns.residplot(x=X_test_prediction, y=residuals_test)
plt.title("Residual analysis of test data")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.show()
'''As there is no visible pattern in the residual plots and the residuals appear to be randomly 
distributed, it suggests that the model's assumptions are generally being met'''
print("As indicated by the various evaluating metrics values like very high r2 score etc and randomly distributed "
      "residual plots.")
print("Our model is very much accurate for predicting car sales amount")

