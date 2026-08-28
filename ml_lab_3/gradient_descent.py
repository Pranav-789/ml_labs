import numpy as np

class GDRegressor:
    def __init__(self, lr = 0.01, epochs = 100):
        self.coef_ = None
        self.intercept_ = None
        self.lr = lr
        self.epochs = epochs

    def fit(self, X_train, y_train):
        self.intercept_ = 0
        self.coef_ = np.ones(X_train.shape[1]) # array of size shape[1] all initialised to 1

        for i in range(self.epochs):
            y_hat = np.dot(X_train, self.coef_) + self.intercept_
            intercept_der = -2 * np.mean(y_train - y_hat)
            self.intercept_ = self.intercept_ - (self.lr*intercept_der)
            coef_der = -2 * np.dot((y_train - y_hat), X_train)/X_train.shape[0]
            self.coef_ = self.coef_ - (self.lr * coef_der)
            
    def predict(self, X_test):
        return np.dot(X_test, self.coef_) + self.intercept_
    

class SGDRegressor:
    def __init__(self, lr = 0.01, epochs = 100):
        self.coef_ = None
        self.intercept_ = None
        self.lr = lr
        self.epochs = epochs

    def fit(self, X_train, y_train):
        self.intercept_ = 0
        self.coef_ = np.ones(X_train.shape[1]) # array of size shape[1] all initialised to 1

        for i in range(self.epochs):
            for j in range(X_train.shape[0]): # in every epoch, row wise calc
                idx = np.random.randint(0, X_train.shape[0]) # randomly selected index
                y_hat = np.dot(X_train[idx], self.coef_) + self.intercept_
                intercept_der = -2 * (y_train[idx] - y_hat)
                self.intercept_ = self.intercept_ - (self.lr * intercept_der)
                coef_der = -2*((y_train[idx]-y_hat)*X_train[idx])
                self.coef_ = self.coef_ - (self.lr*coef_der)
                pass
            
    def predict(self, X_test):
        return np.dot(X_test, self.coef_) + self.intercept_

    
if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    # Generate dataset
    X, y = make_regression(
        n_samples=1000,
        n_features=5,
        noise=10,
        random_state=42
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Your model
    gd = SGDRegressor(lr=0.01, epochs=1000)
    gd.fit(X_train, y_train)
    y_pred_gd = gd.predict(X_test)

    # Sklearn model
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    # Compare
    print("===== Coefficients =====")
    print("Gradient Descent :", gd.coef_)
    print("Sklearn          :", lr.coef_)

    print("\n===== Intercept =====")
    print("Gradient Descent :", gd.intercept_)
    print("Sklearn          :", lr.intercept_)

    print("\n===== R² Score =====")
    print("Gradient Descent :", r2_score(y_test, y_pred_gd))
    print("Sklearn          :", r2_score(y_test, y_pred_lr))

    print("\n===== First 5 Predictions =====")
    print("Gradient Descent :", y_pred_gd[:5])
    print("Sklearn          :", y_pred_lr[:5])