import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__( self,
                  epochs: int = 20,
                  learning_rate: float = 0.0001,
                  threshold: float = 0.5):
        """
        Initialize the LinearRegression model.

        Parameters:\n
        - epochs (int): Number of training epochs (default=50).
        - learning_rate (float): Learning rate for gradient descent (default=0.0001).
        - strategy (str): Optimization strategy, either 'SGD' (Stochastic Gradient Descent) or 'NE' (Normal Equation) (default='SGD').
        - regularization: A tuple that specifies the regularization type and strength. It consists of two values: The regularization type, which can be one of 'l1' (Lasso), 'l2' (Ridge), 'eln' (Elastic Net), or None for no regularization. The regularization strength (λ).
        """
        self.params = {
            'ep': epochs,
            'lr': learning_rate,
            'th': threshold,
            'wi': None,
            'b': None,
            'stra': ''
        }

    def fit( self, X, y ):
        if self.params['wi'] is None:
            num_features = 1 if isinstance(X[ 0 ], (float, int)) else len(X[ 0 ])
            # initialize random parameters
            self.params[ 'wi' ] = np.random.uniform(0, 1, size=num_features) * 1
            self.params[ 'b' ] = np.random.uniform(0, 1) * 1
            self.params[ 'stra' ] = 'bin' if len(set(y)) == 2 else 'mul'

        epochs = self.params['ep']
        x = np.array(X)
        y = np.array(y)

        for epoch in range(epochs):
            predictions = self.predict(X)
            gradients = self.__backward_propagation(x, y, predictions)
            self.__update_params(gradients)
            print(f"Epoch {epoch + 1}: \tLoss {self.cost_function(predictions, y):.2f}")

    def predict( self, X ):
        w = self.params[ 'wi' ]
        b = self.params[ 'b' ]
        threshold = self.params['th']
        strategy = self.params['stra']
        if w is not None and b is not None:
            x = np.array(X)
            z = x @ w + b
            if strategy == 'bin':
                # Sigmoid
                predictions = np.array([1 if self.__sigmoid(i) >= threshold else 0 for i in z])
            else:
                # Softmax
                predictions = np.array([self.__softmax(z, k) for k in range(x.shape[0]) ])
            return predictions
        raise Exception('Model not fitted')

    def cost_function( self, predictions, y ):
        e = 1e-15  # Small epsilon value to avoid taking the log of zero
        ones = np.ones(len(y))
        cost = -(np.sum(y * np.log(predictions + e) + (ones - y) * np.log(ones - predictions + e)))
        return cost

    def log_likelihood_function( self, predictions, y ):
        return -1 * self.cost_function(predictions, y)

    def __backward_propagation( self, x, y, predictions ):
        gradients = {}
        df = (y - predictions) * -1
        dw = np.array([np.sum(np.multiply(df, x[:, col])) for col in range(x.shape[1])])
        db = np.sum(df)
        gradients[ 'dw' ] = dw
        gradients[ 'db' ] = db
        return gradients

    def __sigmoid( self, z ):
        return 1 / (1 + np.exp(-z))

    def __softmax( self, z, k):
        return np.exp(z[k]) / np.sum(np.exp(z))

    def __update_params( self, gradients ):
        grad_b = gradients[ 'db' ]
        grad_w = gradients[ 'dw' ]
        lr = self.params[ 'lr' ]
        w = self.params[ 'wi' ]
        b = self.params[ 'b' ]
        self.params[ 'b' ] = b - lr * grad_b
        self.params[ 'wi' ] = w - lr * grad_w

