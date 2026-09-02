import numpy as np

class GDRegressor:
    def __init__(self, lr=0.01, epochs=100):
        self.coef_ = None
        self.intercept_ = None
        self.lr = lr
        self.epochs = 1000
        self.coef_history_ = []
        self.intercept_history_ = []

    def fit(self, X_train, y_train):
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        self.intercept_ = 0.0
        self.coef_ = np.zeros(X_train.shape[1])

        self.coef_history_ = [self.coef_.copy()]
        self.intercept_history_ = [self.intercept_]

        for i in range(self.epochs):
            y_hat = np.dot(X_train, self.coef_) + self.intercept_
            intercept_der = -2 * np.mean(y_train - y_hat)
            self.intercept_ = self.intercept_ - (self.lr * intercept_der)
            coef_der = -2 * np.dot((y_train - y_hat), X_train) / X_train.shape[0]
            self.coef_ = self.coef_ - (self.lr * coef_der)

            self.coef_history_.append(self.coef_.copy())
            self.intercept_history_.append(self.intercept_)
            
    def predict(self, X_test):
        X_test = np.asarray(X_test)
        return np.dot(X_test, self.coef_) + self.intercept_

class MiniBatchGDRegressor:
    def __init__(self, batch_size=35, lr=0.01, epochs=100):
        self.coef_ = None
        self.intercept_ = None
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = 1000
        self.coef_history_ = []
        self.intercept_history_ = []

    def fit(self, X_train, y_train, batch_size=None, lr=None, epochs=None):
        if batch_size is not None:
            self.batch_size = batch_size
        if lr is not None:
            self.lr = lr
        if epochs is not None:
            self.epochs = epochs

        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)

        self.intercept_ = 0.0
        self.coef_ = np.zeros(X_train.shape[1])

        self.coef_history_ = [self.coef_.copy()]
        self.intercept_history_ = [self.intercept_]

        n_samples = X_train.shape[0]

        for i in range(self.epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size

                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                y_hat = np.dot(X_batch, self.coef_) + self.intercept_

                intercept_der = -2 * np.mean(y_batch - y_hat)
                coef_der = -2 * np.dot(y_batch - y_hat, X_batch) / X_batch.shape[0]

                self.intercept_ -= self.lr * intercept_der
                self.coef_ -= self.lr * coef_der

            self.coef_history_.append(self.coef_.copy())
            self.intercept_history_.append(self.intercept_)

    def predict(self, X_test):
        X_test = np.asarray(X_test)
        return np.dot(X_test, self.coef_) + self.intercept_

class SGDRegressor:
    def __init__(self, lr=0.01, epochs=100):
        self.coef_ = None
        self.intercept_ = None
        self.lr = lr
        self.epochs = 1000
        self.coef_history_ = []
        self.intercept_history_ = []

    def fit(self, X_train, y_train, epochs=None, lr=None):
        if lr is not None:
            self.lr = lr
        if epochs is not None:
            self.epochs = epochs

        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)

        self.intercept_ = 0.0
        self.coef_ = np.zeros(X_train.shape[1])

        self.coef_history_ = [self.coef_.copy()]
        self.intercept_history_ = [self.intercept_]

        for i in range(self.epochs):
            for j in range(X_train.shape[0]):
                idx = np.random.randint(0, X_train.shape[0])

                y_hat = np.dot(X_train[idx], self.coef_) + self.intercept_
                intercept_der = -2 * (y_train[idx] - y_hat)
                coef_der = -2 * ((y_train[idx] - y_hat) * X_train[idx])

                self.intercept_ -= self.lr * intercept_der
                self.coef_ -= self.lr * coef_der

            self.coef_history_.append(self.coef_.copy())
            self.intercept_history_.append(self.intercept_)

    def predict(self, X_test):
        X_test = np.asarray(X_test)
        return np.dot(X_test, self.coef_) + self.intercept_

