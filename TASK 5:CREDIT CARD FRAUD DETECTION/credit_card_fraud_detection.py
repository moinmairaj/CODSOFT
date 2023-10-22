import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
card_data = pd.read_csv(r"/Users/moinmairaj/Desktop/Data Science/creditcard.csv")
print(card_data.head())
print(card_data.info())
# all the columns are numeric in nature
print(card_data.isnull().values.any())
# no null value also
print(card_data.duplicated().any())
card_data = card_data.drop_duplicates()
print(card_data.duplicated().any())
# handled duplicates also
print(card_data.info())
# since we can train all the columns in the model so no need to drop any
X = card_data.drop(columns=['Class'])
Y = card_data['Class']
scalar = StandardScaler()
X = scalar.fit_transform(X)
print(X)
print(Y.head())
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
model = LogisticRegression()
model.fit(X_train, Y_train)
X_train_predict = model.predict(X_train)
X_test_predict = model.predict(X_test)
train_pred_score = accuracy_score(Y_train, X_train_predict)
test_pred_score = accuracy_score(Y_test, X_test_predict)
print(train_pred_score, test_pred_score)
# these results suggest that our logistic regression model is performing well.