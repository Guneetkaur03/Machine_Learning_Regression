import pandas as pd
from pandas import DataFrame
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

data = pd.read_csv('winequality-red.csv')
df = DataFrame(data,columns=['pH','density','alcohol','quality'])

X = df[['pH','density','alcohol']].astype(float) # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
y = df['quality'].astype(float)

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

y_pred = regr.predict(X_test)
print('Variance score: %.2f' % r2_score(y_test, y_pred))
