import Perceptron as prcp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# data from https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
df = pd.read_csv('data\iris.data.csv' ,header=None)
print(df.tail())

Y = df.iloc[0:100,4].values # get classification array
Y = np.where(Y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0,2]].values # take 1st and 3rd feature as input array

setosa = X[Y == -1]
versicolor = X[Y == 1]
plt.plot(setosa[:,0],setosa[:,1],'r^', label='setosa')
plt.plot(versicolor[:,0],versicolor[:,1],'gx',label='versicolor')

plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')



linear_c = prcp.LinearClassification()
linear_c.fit(X,Y)
W = linear_c.W
X = range(0,10)
Y = -(W[0] + W[1]*X)/W[2]
plt.plot(X,Y,'k-')



plt.show()