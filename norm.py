"""data distribution with normal distribution over all the columns of data """
import scipy
import pandas as pd
import numpy as np
import seaborn as sns
import seaborn as sns
import matplotlib.pyplot as plt


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


for n in range(4):
    dataL1=D[:,n]
    fig1, ax1 = plt.subplots()
    plt.title('data column %d'%n)
    for i in range(3):
        class_ind=np.where(np.array(classLabels)==i)
        mean, var  = scipy.stats.distributions.norm.fit(dataL1[class_ind])
        x = np.linspace(dataL1.min(),dataL1.max(),50)
        fitted_data = scipy.stats.distributions.norm.pdf(x, mean, var)
        ax1 = plt.hist(dataL1, density=True)
        ax1 = plt.plot(x,fitted_data,'r-')
        print(mean,var)
