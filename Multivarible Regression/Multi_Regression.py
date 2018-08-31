import pandas as pd
from pandas import DataFrame
from matplotlib import pyplot as plt
from sklearn import linear_model
import statsmodels.api as sm
from mpl_toolkits import mplot3d
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('winequality-red.csv')
df = DataFrame(data,columns=['pH','density','alcohol','quality'])

X = df[['pH','density','alcohol']].astype(float) # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
Y = df['quality'].astype(float)

regr = linear_model.LinearRegression()
regr.fit(X, Y)
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

a = regr.predict(X)
print('Variance score: %.2f' % r2_score(z, a))
