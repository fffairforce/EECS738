from scipy.stats import mode
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal


class GMM:
    def __init__(self, k, max_iter):
        self.k = k
        self.max_iter = int(max_iter)

    def init(self, X):
        self.shape = X.shape
        self.n, self.m = self.shape
        self.phi = np.full(shape=self.shape, fill_value=1/self.k)
        self.weights = np.full(shape=self.shape, fill_value=1/self.k)
        random_row = np.random.randint(low=0, high=self.n, size=self.k)
        self.mu = [X[row_ind, :] for row_ind in random_row]
        self.sigma = [np.cov(X.T) for _ in range(self.k)]

    def stepE(self, X):
        self.weights = self.predict_prob(X)
        self.phi = self.weights.mean(axis=0)

    def stepM(self, X):
        for i in range(self.k):
            weight = self.weights[:, [i]]
            tot_weight = weight.sum()
            self.mu[i] = (X*weight).sum(axis=0)/tot_weight
            self.sigma[i] = np.cov(X.T, aweights=(weight/tot_weight).flatten(), bias=True)

    def predict_prob(self, X):
        likelihood = np.zeros((self.n, self.k))
        for i in range(self.k):
            distribution = multivariate_normal(mean=self.mu[i], cov=self.sigma[i])
            likelihood[:, i] = distribution.pdf(X)  # Probability density function
            numerator = likelihood * self.phi
            denominator = numerator.sum(axis=1)[:, np.newaxis]
            weights = numerator / denominator
            return weights

    def fit(self, X):
        self.init(X)
        for iteration in range(self.max_iter):
            self.stepE(X)
            self.stepM(X)

    def predict(self, X):
        weights = self.predict_prob(X)
        return np.argmax(weights, axis=1)


def readData(filename):
    data = pd.read_csv(filename, names=['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width', 'Species'])
    dataframe = data.drop(columns='Species')
    dataMat = dataframe.to_numpy()
    # add labels matrix
    classLabels = []
    for line in range(len(data)-1):
        if data.iloc[line][4] == 'Iris-setosa':
            classLabels.append(1)
        elif data.iloc[line][4] == 'Iris-versicolor':
            classLabels.append(2)
        elif data.iloc[line][4] == 'Iris-virginica':
            classLabels.append(3)
    return dataMat, classLabels


def jitter(x):
    return x + np.random.uniform(low=-0.15, high=0.15, size=x.shape)


def plot_axis_pairs(data, axis_pairs, clusters, classes):
    feature_names = ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width', 'Species']
    n_rows = len(axis_pairs) // 2
    n_cols = 2
    plt.figure(figsize=(16, 10))
    for index, (x_axis, y_axis) in enumerate(axis_pairs):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.title('GMM Clusters')
        plt.xlabel(feature_names[x_axis])
        plt.ylabel(feature_names[y_axis])
        plt.scatter(
            jitter(data[:, x_axis]),
            jitter(data[:, y_axis]),
            # c=clusters,
            cmap=plt.cm.get_cmap('brg'),
            marker='x')
    plt.tight_layout()


if __name__ == '__main__':
    filename = 'iris.data'
    data, classLabels = readData(filename)
    np.random.seed(40)
    gmm = GMM(k=3, max_iter=10)
    gmm.fit(data)
    permutation = np.array([
        mode(classLabels[gmm.predict(data) == i]).mode.item()
        for i in range(gmm.k)])
    permuted_prediction = permutation[gmm.predict(data)]
    print(np.mean(classLabels == permuted_prediction))
    confusion_matrix(classLabels, permuted_prediction)
    plot_axis_pairs(data=data,
        axis_pairs=[
            (0, 1), (2, 3),
            (0, 2), (1, 3)],
        clusters=permuted_prediction,
        classes=classLabels)

#q to ask
# 1. supervised/unsupervised
# 2. how would you want the model mixed
# 3. what output do you expect to see, visualized cluster? R2?
# 4. are we doing prediction already?
