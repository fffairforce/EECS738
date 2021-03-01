"""binom"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom


data = pd.read_csv('iris.data', names=['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width', 'Species'])
dataMat = data.drop(columns='Species')
D = dataMat.to_numpy()

classLabels = []
for line in range(len(data)):
    if data.iloc[line][4] == 'Iris-setosa':
        classLabels.append(0)
    elif data.iloc[line][4] == 'Iris-versicolor':
        classLabels.append(1)
    elif data.iloc[line][4] == 'Iris-virginica':
        classLabels.append(2)

dataL1=D[:,0]
fig, ax = plt.subplots(1, 1)
class_ind=np.where(np.array(classLabels)==0)
class_data = dataL1[class_ind]
n, p = 50, 1/3
r_values = list(range(n+1))
mean, var = binom.stats(n, p)
dist = [binom.pmf(r, n, p) for r in r_values ]
ax = plt.bar(r_values, dist)


#fig2, ax2 = plt.subplots(1, 1)
class_ind=np.where(np.array(classLabels)==1)
class_data = dataL1[class_ind]
n, p = 100, 2/3
r_values = list(range(n+1))
mean, var = binom.stats(n, p)
dist = [binom.pmf(r, n, p) for r in r_values ]
ax1 = plt.bar(r_values, dist)
plt.title('binomial with chosen class vs rest class')
plt.show()
