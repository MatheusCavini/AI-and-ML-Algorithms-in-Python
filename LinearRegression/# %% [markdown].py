# %% [markdown]
# # Implementação do Zero

# %%
import numpy as np

# %%
class LinearRegression:
    def __init__(self, lr = 0.001, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None

    def fit(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        X_train = np.concatenate([X_train, np.ones([n_samples,1])], axis=1)
        self.weights =  np.zeros(n_features+1)

        for n in range(self.n_iter):
            y_pred = np.dot(X_train, self.weights)

            gradient = (1/n_samples) * ((y_pred-y_train).T @ X_train)

            self.weights -= (self.lr * gradient)

        return self.weights


    def predict(self, X_test):
        n_samples, n_features = X_test.shape
        X_test = np.concatenate([X_test, np.ones([n_samples,1])], axis=1)
        y_pred = np.dot(X_test, self.weights)
        return y_pred



# %% [markdown]
# # Teste

# %%
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

X, y = datasets.make_regression(n_samples=100, n_features=2, noise=20, random_state=4)
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size = 0.2)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:, 0], X[:, 1], y, color='b', marker='o', s=30)
plt.show()

# %%
regressor = LinearRegression(0.01, 1500)
weights = regressor.fit(X_train, y_train)
print(weights)
predictions = regressor.predict(X_test)

# %%
def mse(y_test, predictions):
    return np.mean((y_test-predictions)**2)

mse =  mse(y_test, predictions)
print(mse)

# %%
y_pred_line =  regressor.predict(X)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:, 0], X[:, 1], y, color='b', marker='o', s=30)
ax.scatter(X[:,0], X[:,1], y_pred_line, color='r')
plt.show()



