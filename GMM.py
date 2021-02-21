import numpy as np
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
        for iter in range(self.max_iter):
            self.stepE(X)
            self.stepM(X)