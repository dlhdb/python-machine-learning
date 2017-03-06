import numpy as np

class LinearClassification(object):
    def __init__(self, eta=0.01, n_iter=10):
        self.eta=eta
        self.n_iter=n_iter

    def fit(self, X, Y):
        self.W = np.zeros(1+X.shape[1])

        for _ in range(self.n_iter):
            errorCnt = 0
            for xi, yi in zip(X,Y):
                error = yi - self.predict(xi)
                self.W[1:] += self.eta * (error) * xi
                self.W[0] += self.eta * (error)
                errorCnt += int(error != 0.0)
            print(errorCnt)
        return self

    def net_input(self, X):
        return np.dot(X, self.W[1:]) + self.W[0]

    def predict(self, X):
        return np.where(self.net_input(X) > 0.0, 1, -1)