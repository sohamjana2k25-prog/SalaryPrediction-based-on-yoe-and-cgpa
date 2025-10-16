import numpy as np


class linear_regression:
    def __init__(self, l_rate, no_iter):
        self.l_rate = l_rate
        self.no_iter = no_iter
        self.w = None
        self.b = None

    def fit(self, X_train, Y_train):
        self.w = np.ones(X_train.shape[1])
        self.b = 0

        for i in range(self.no_iter):
            for j in range(X_train.shape[0]):
                idx = np.random.randint(X_train.shape[0])
                y_pred = np.dot(X_train[idx], self.w) + self.b
                dw = -2 * np.dot(Y_train[idx] - y_pred, X_train[idx])
                db = -2 * (Y_train[idx] - y_pred)
                self.w -= self.l_rate * dw
                self.b -= self.l_rate * db

    def predict(self, X):
        return np.dot(X, self.w) + self.b
