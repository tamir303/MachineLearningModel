import numpy as np
import matplotlib.pyplot as plt


class LWLR:
    # Weights_Vector(theta) = inv(X.T*W*X) * (X.T*W*Y)
    def __init__( self,
                  tau=0.01 ):
        self.params = {
            'tau': tau,
        }

    def fit_and_plot( self, X, y ):
        if X.shape[1] > 1:
            raise ValueError('Cannot plot more than 2-D features')
        X_linspace = np.linspace(np.min(X), np.max(X), X.shape[ 0 ])
        y_pred = np.array([self.__predict(X, np.c_[y], x) for x in X_linspace])
        plt.style.use('seaborn')
        plt.title("The scatter plot for the value of tau = %.5f" % self.params[ 'tau' ])
        plt.scatter(X, y, color='red')
        plt.scatter(X_linspace, y_pred, color='green')
        plt.show()

    def fit_and_predict( self, X, y ):
        y_pred = np.array([ self.__predict(X, np.c_[y], x) for x in X])

        return y_pred

    def __predict( self, X, y, xi ):
        n_samples = X.shape[ 0 ]
        q = np.array([xi] + [1])
        X = np.hstack((X, np.ones((n_samples, 1))))
        W = self.__kernel_weights(X, xi)
        theta = np.linalg.pinv(X.T @ W @ X) * (X.T @ W @ y)
        pred = np.dot(q, theta).tolist()

        return pred

    def __kernel_weights( self, X, xi ):
        n_samples = X.shape[ 0 ]
        W = np.mat(np.eye(n_samples))
        sq_tau = self.params[ 'tau' ] ** 2
        for idx in range(n_samples):
            W[ idx, idx ] = np.exp(-np.dot(X[ idx ] - xi, (X[ idx ] - xi).T) / (2 * sq_tau))

        return W
