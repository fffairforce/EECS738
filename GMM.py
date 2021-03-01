import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def readData(filename):
    data = pd.read_csv(filename, names=['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width', 'Species'])
    dataframe = data.drop(columns='Species')
    dataMat = dataframe.to_numpy()
    # add labels matrix
    classLabels = []
    for line in range(len(data)):
        if data.iloc[line][4] == 'Iris-setosa':
            classLabels.append(1)
        elif data.iloc[line][4] == 'Iris-versicolor':
            classLabels.append(2)
        elif data.iloc[line][4] == 'Iris-virginica':
            classLabels.append(3)

    return dataMat, classLabels


def readwineData(filename):
    data = pd.read_csv('wine.data', names=['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline'])
    dataMat = data.to_numpy()
    #make classlabel
    class1 = np.zeros((59,1))
    class2 = np.ones((71,1))
    class3 = 2*np.ones((48,1))
    classLabels = np.append(class1,class2,axis=0)
    classLabels = np.append(classLabels,class3, axis=0)
    return dataMat, classLabels


class GMM:

    # create array r for each datapoint xi the probability r[i] that datapoint xi belongs to cluster c
    def __init__(self, col, k, X, iterations):
        # input col-data column to analyze, k-classes, X-dataMat, iteration-iterations to operate
        self.col = col
        self.k = k
        self.iterations = iterations
        self.X = X
        self.mu = None
        self.pi = None
        self.var = None
        self.y = np.zeros(150)  # data scatter plot base line

    def fit_run(self):
        # now suedo assign random initial values, consider used seed() / random later
        self.n = self.X.shape
        random_row = np.random.uniform(low=self.X.min(), high=self.X.max(), size=self.k)
        self.mu = random_row
        self.pi = np.full(shape=self.k, fill_value=1 / self.k)  # []
        self.var = np.full(shape=self.k, fill_value=1 / self.k)  # [.6,.4,.3]

        # step-E
        for iter in range(self.iterations):
            r = np.zeros((len(dataL1), 3))
            # Probability for each datapoint x_i to belong to gaussian g
            for c, g, p in zip(range(3), [norm(loc=self.mu[0], scale=self.var[0]),
                                          norm(loc=self.mu[1], scale=self.var[1]),
                                          norm(loc=self.mu[2], scale=self.var[2])], self.pi):
                #         classLabels = np.array(classLabels)
                #         ind = np.where(classLabels==c)
                r[:, c] = p * g.pdf(self.X)
            # Normalize the probabilities such that each row of r sums to 1 and weight it by mu_c == the fraction of points belonging to cluster c
            for i in range(len(r)):
                r[i] = r[i] / (np.sum(self.pi) * np.sum(r, axis=1)[i])

            # print(np.shape(r))
            # np.shape(dataL1)
            # data plot
            fig1, ax1 = plt.subplots()
            ax1.set_title('data column %d , iter: %d' % (self.col, iter))
            ax1 = plt.scatter(self.X, self.y, c=classLabels, edgecolors='none')
            # plot gaussians
            for g, c in zip(
                    [norm(loc=self.mu[0], scale=self.var[0]).pdf(np.linspace(self.X.min(), self.X.max(), num=150)),
                     norm(loc=self.mu[1], scale=self.var[1]).pdf(np.linspace(self.X.min(), self.X.max(), num=150)),
                     norm(loc=self.mu[2], scale=self.var[2]).pdf(np.linspace(self.X.min(), self.X.max(), num=150))],
                    ['r', 'g', 'b']):
                ax1 = plt.plot(np.linspace(self.X.min(), self.X.max(), num=150), g, c=c)

            # step-M
            # calculate m_c
            m_c = []
            for c in range(len(r[0])):
                m = np.sum(r[:, c])
                m_c.append(m)
            # calculate pi_c/For each cluster c, calculate the fraction of points pi_c
            for k in range(len(m_c)):
                self.pi[k] = (m_c[k] / np.sum(m_c))
            # calculate mu_c
            self.mu = np.sum(self.X.reshape(len(self.X), 1) * r, axis=0) / m_c
            # calculate var_c
            var_c = []
            for c in range(len(r[0])):
                var_c.append((1 / m_c[c]) * np.dot(
                    ((np.array(r[:, c]).reshape(len(self.X), 1)) * (self.X.reshape(len(self.X), 1) - self.mu[c])).T,
                    (self.X.reshape(len(self.X), 1) - self.mu[c])))

            plt.show()


if __name__ == "__main__":
    # if iris
    filename = 'iris.data'
    data, classLabels = readData(filename)
    # if wine
    # filename = 'wine.data'
    # data, classLabels = readwineData(filename)
    n, m = np.shape(data)
    for i in range(m):
        dataL1 = data[:, i]
        np.random.seed(i)
        gmm = GMM(i, 3, dataL1, 10)
        gmm.fit_run()
