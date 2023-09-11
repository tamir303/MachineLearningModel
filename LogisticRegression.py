import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__( self,
                  epochs: int = 100,
                  learning_rate: float = 0.01,
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
            'w': None,
            'b': None,
            'stra': ''
        }

    def fit( self, X, y, verbose=True ):
        def binomial():
            self.params[ 'w' ] = np.random.normal(loc=0.0, scale=std, size=n_features) * 1
            self.params[ 'b' ] = np.random.normal(loc=0.0, scale=std) * 1

        def multinomial():
            self.params[ 'w' ] = np.random.normal(loc=0.0, scale=std, size=(n_features, n_classes)) * 1
            self.params[ 'b' ] = np.random.normal(loc=0.0, scale=std, size=n_classes) * 1

        if self.params['w'] is None:
            std = 1e-3
            n_features = X.shape[ 1 ]  # number of features
            n_classes = len(np.unique(y))  # number of classes in the dataset
            strategy = self.params[ 'stra' ] = 'bin' if n_classes == 2 else 'mul'
            binomial() if strategy == 'bin' else multinomial()

        epochs = self.params['ep']
        x = np.array(X)
        y = np.array(y)

        for epoch in range(epochs):
            predictions = self.predict(x, prob=True)
            gradients = self.__backward_propagation(x, y, predictions)
            self.__update_params(gradients)
            if verbose:
                print(f"Epoch {epoch + 1}: \tLoss {self.cost_function(predictions, y):.2f}")

    def predict( self, X, prob=False ):
        w = self.params[ 'w' ]
        b = self.params[ 'b' ]
        if w is not None and b is not None:
            x = np.array(X)
            z = x @ w + b
            return self.__predictor(z, prob)
        raise Exception('Model not fitted')

    def __backward_propagation( self, x, y, predictions ):
        gradients = {}
        strategy = self.params['stra']
        n_samples = y.shape[ 0 ]
        dscores = predictions.copy()

        if strategy == 'bin':
            df = (y - dscores) * -1
            dw = np.array([ np.sum(np.multiply(df, x[ :, col ])) for col in range(x.shape[ 1 ]) ])
            db = np.sum(df)
        else:
            dscores[ np.arange(n_samples), y ] -= 1
            dscores /= n_samples
            dw = x.T.dot(dscores)
            db = np.sum(dscores, axis=0, keepdims=True)

        gradients[ 'dw' ] = dw
        gradients[ 'db' ] = db
        return gradients

    def __predictor( self, z, prob ):
        def sigmoid():
            return 1 / (1 + np.exp(-z))

        def softmax():
            exp_scores = np.exp(z - np.max(z, axis=1, keepdims=True))
            return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        strategy = self.params['stra']
        if strategy == 'bin':
            score = sigmoid()
            return (score >= self.params['th']).astype(int) if not prob else score
        else:
            score = softmax()
            return np.argmax(score, axis=1) if not prob else score

    def cost_function( self, predictions, y ):
        strategy = self.params[ 'stra' ]
        n_samples = y.shape[ 0 ]  # number of samples
        if strategy == 'bin':
            e = 1e-15  # Small epsilon value to avoid taking the log of zero
            ones = np.ones(n_samples)
            cost = -(np.sum(y * np.log(predictions + e) + (ones - y) * np.log(ones - predictions + e))) / n_samples
        else:
            corr_logprobs = -np.log(predictions[ np.arange(n_samples), y ])
            cost = np.sum(corr_logprobs)

        return cost

    def __update_params( self, gradients ):
        grad_b = gradients[ 'db' ]
        grad_w = gradients[ 'dw' ]
        lr = self.params[ 'lr' ]
        w = self.params[ 'w' ]
        b = self.params[ 'b' ]
        self.params[ 'b' ] = b - lr * grad_b
        self.params[ 'w' ] = w - lr * grad_w

    def plot2D( self, X, y ):
        if X.shape[ 1 ] != 2:
            raise ValueError("Plotting is only supported for 2D feature data.")

        # Generate a grid of points for plotting the decision boundary
        x_min, x_max = X[ :, 0 ].min() - 1, X[ :, 0 ].max() + 1
        y_min, y_max = X[ :, 1 ].min() - 1, X[ :, 1 ].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

        # Predict the class labels for the grid points
        grid_points = np.c_[ xx.ravel(), yy.ravel() ]
        Z = self.predict(grid_points)

        # Reshape the predictions to the meshgrid shape
        Z = Z.reshape(xx.shape)

        # Create a contour plot to visualize the decision boundary
        plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)
        plt.scatter(X[ :, 0 ], X[ :, 1 ], c=y, cmap=plt.cm.RdBu, edgecolors='k')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Decision Boundary')
        plt.show()
