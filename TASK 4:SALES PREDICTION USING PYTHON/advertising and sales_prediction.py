import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
ad_data = pd.read_csv(r'/Users/moinmairaj/Desktop/Data Science/advertising.csv')
print(ad_data.head())
print(ad_data.shape)
print(ad_data.info())
print(ad_data.isnull().values.any())
# no null value present also all columns numeric in nature so data is already pre processed.
# data visualization
print(ad_data.describe())
sns.barplot(x='TV', y='Sales', data=ad_data)
plt.show()
# not clear visualize
# use advanced techniques
sns.scatterplot(x='TV', y='Sales', data=ad_data)
plt.show()
sns.jointplot(x='TV', y='Sales', data=ad_data, kind='hex')
plt.show()
sns.lmplot(x='TV', y='Sales', data=ad_data)
plt.show()
# we see higher TV advertising higher is the sales
# visulaize other columns vs data
sns.scatterplot(x='Radio', y='Sales', data=ad_data)
plt.show()
# effect of TV is more than radio as higher radio does not always mean more sales
sns.scatterplot(x='Newspaper', y='Sales', data=ad_data)
plt.show()
# from the plot newspaper has a negative impact on the sales
# higher newspaper advertisement lower is the sales
# ,so we conclude higher sales are due to high TV advertisement, lower Newspaper advertisement and moderate Radio


# data division and splitting
X = ad_data.drop(columns=['Sales'], axis=1)
Y = ad_data['Sales']
# scaling X
scalar = StandardScaler()
X = scalar.fit_transform(X)
print(X)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=4)
print(X.shape, X_train.shape, X_test.shape)
print(Y.shape, Y_train.shape, Y_test.shape)
# modeling
model = LinearRegression()
model.fit(X_train, Y_train)
# predict sales for test and train sets
X_train_prediction = model.predict(X_train)
X_test_prediction = model.predict(X_test)
# accuracy check
mse_train = mean_squared_error(Y_train, X_train_prediction)
rmse_train = np.sqrt(mse_train)
mae_train = mean_absolute_error(Y_train, X_train_prediction)
r2_train = r2_score(Y_train, X_train_prediction)
mse_test = mean_squared_error(Y_test, X_test_prediction)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(Y_test, X_test_prediction)
r2_test = r2_score(Y_test, X_test_prediction)
print("Metric:       ", " Train", " "*16, "Test")
print("1.MSE:   ", mse_train, " "*3, mse_test)
print("2.RMSE:  ", rmse_train, " "*3, rmse_test)
print("3.MAE:   ", mae_train, " "*3, mae_test)
print("4.R2:    ", r2_train, " "*3, r2_test)
print("Values obtained for the various metrics indicates that the model performance is good.")
# Residual analysis
# training data
residuals_train = Y_train - X_train_prediction
plt.figure(figsize=(8, 6))
sns.residplot(x=X_train_prediction, y=residuals_train)
plt.title('Residual Plot Training Data')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()
# test data analysis
residuals_test = Y_test - X_test_prediction
plt.figure(figsize=(8,6))
sns.residplot(x=X_test_prediction, y=residuals_test)
plt.title("Residual Plot Test Data")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.show()
'''Residual Plot Analysis: As there is no visible pattern in the residual plots and the residuals appear to be randomly 
distributed, it suggests that the model's assumptions are generally being met. The random distribution of residuals 
within a consistent range (-4 to 4 for training and -4 to 3 for testing) indicates that the model is capturing 
the underlying patterns well and is not biased towards specific ranges of the target variable. 
This is a positive indication for the model's effectiveness.'''
print("Based on the residual plot analysis and the evaluation of various metrics, "
      "the model seems to perform well, with relatively low error rates and high R2 values.")
print("It appears to be effectively capturing the patterns in the data "
      "and making accurate predictions on both the training and test datasets.")