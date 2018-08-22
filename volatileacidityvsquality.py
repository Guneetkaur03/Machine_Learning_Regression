import pandas as pd
import numpy as np
from cmath import sqrt
import scipy.stats as stats
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv('winequality-red.csv')

X = data['volatile acidity'].values[:,np.newaxis]
y = data['quality'].values

xtotal=0
ytotal =0
xytotal =0
x2total =0
y2total =0
n =1599

for i in range(n):
    xtotal +=X[i]

for i in range(n):
    ytotal +=y[i]

for i in range(n):
    xytotal +=(X[i]*y[i])
    
for i in range(n):
    x2total +=(X[i]*X[i])
    
for i in range(n):
    y2total +=(y[i]*y[i])


num = xytotal - ((xtotal*ytotal)/n)

den1 = (x2total - ((xtotal*xtotal)/n))

den2 = (y2total - ((ytotal*ytotal)/n))

den = den1*den2
denf = sqrt(den)
r = num/denf
print("Correlation for volatile acidity and quality of wine")
print(r)

model = LinearRegression()

model.fit(X, y)
plt.scatter(X, y,color='blue')
plt.plot(X, model.predict(X),color='k')
plt.title('quality Vs volatile acid ', fontsize=14)
plt.xlabel('volatile acid', fontsize=14)
plt.ylabel('quality', fontsize=14)
plt.show()
print('Intercept: \n', model.intercept_)
print('Coefficients: \n', model.coef_)
