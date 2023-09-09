import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__( self,
                  epochs: int = 20,
                  learning_rate: float = 0.0001,
                  strategy: str = 'SGD',
                  regularization: tuple[ str, float ] = None ):
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
            'stra': strategy,
            'regu': regularization,
            'wi': None,
            'b': None
        }

    def fit( self, X, y ):
        """
        Fit the linear regression model to the given training data.

        Parameters:
        - X (numpy.ndarray): Input features (independent variables).
        - y (numpy.ndarray): Target values (dependent variable).
        """
        self.__validate_params(X, y)

        def SGD():
            if self.params['wi'] is None:
                num_features = 1 if isinstance(X[ 0 ], (float, int)) else len(X[ 0 ])
                # initialize random parameters
                self.params[ 'wi' ] = np.random.uniform(0, 1, size=num_features) * -1
                self.params[ 'b' ] = np.random.uniform(0, 1) * -1

            epochs = self.params[ 'ep' ]
            for epoch in range(epochs):
                predictions = self.predict(X)
                gradients = self.__backward_propagation(X, y, predictions)
                self.update_params(gradients)
                print(f"Epoch {epoch + 1}: \tLoss {self.cost_function(predictions, y):.2f}")

        def NE():
            X_with_intercept = np.column_stack((np.ones(len(X)), X))
            coefficients = np.linalg.pinv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
            self.params[ 'b' ] = coefficients[ 0 ]
            self.params[ 'wi' ] = coefficients[ 1: ]
            print(f"Loss {self.cost_function(self.predict(X), y):.2f}")

        strategy = self.params[ 'stra' ]
        if strategy == 'SGD':
            SGD()
        elif strategy == 'NE':
            NE()

    def predict( self, X ):
        w = self.params[ 'wi' ]
        b = self.params[ 'b' ]
        if w is not None and b is not None:
            train_input = np.array(X)
            predictions = train_input @ w + b
            return predictions
        raise Exception('Model not fitted')

    def cost_function( self, predictions, y ):
        cost = np.mean((y - predictions) ** 2)

        if self.params[ 'regu' ] is not None:
            reg_type, reg_strength = self.params[ 'regu' ]
            w = self.params[ 'wi' ]
            penalty = None
            if reg_type == 'l1':
                penalty = reg_strength * np.sum(np.abs(w))
            elif reg_type == 'l2':
                penalty = reg_strength * np.sum(w ** 2)
            elif reg_type == 'eln':
                penalty = reg_strength * (0.5 * np.sum(np.abs(w) + 0.5 * np.sum(w ** 2)))

            cost += penalty

        return cost

    def __backward_propagation( self, X, y, predictions ):
        """
        Calculate gradients for the model parameters (weights and bias).
        """
        gradients = {}
        df = (y - predictions) * -1
        train_input = np.array(X)
        dw = np.array([np.mean(np.multiply(df, train_input[:, col])) for col in range(train_input.shape[1])])  # Weights derivative
        db = np.mean(df)  # Bias derivative

        if self.params[ 'regu' ] is not None:
            reg_type, reg_strength = self.params[ 'regu' ]
            w = self.params[ 'wi' ]
            gradient = None
            if reg_type == 'l1':
                gradient = reg_strength * np.sign(w)
            elif reg_type == 'l2':
                gradient = reg_strength * w * 2
            elif reg_type == 'eln':
                gradient = reg_strength * (0.5 * np.abs(w) + w)

            dw += gradient

        gradients[ 'dw' ] = dw
        gradients[ 'db' ] = db
        return gradients

    def update_params( self, gradients ):
        grad_b = gradients[ 'db' ]
        grad_w = gradients[ 'dw' ]
        lr = self.params[ 'lr' ]
        w = self.params[ 'wi' ]
        b = self.params[ 'b' ]
        self.params[ 'b' ] = b - lr * grad_b
        self.params[ 'wi' ] = w - lr * grad_w

    def plot( self, X, y ):
        y_pred = X * self.params[ 'wi' ] + self.params[ 'b' ]
        plt.plot(X, y, '+', label='Actual values')
        plt.plot(X, y_pred, label='Predicted values')
        plt.xlabel('Test input')
        plt.ylabel('Test Output or Predicted output')
        plt.legend()
        plt.show()

    def __validate_params( self, X, y ):
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y must be NumPy arrays.")

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of data points.")

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)  # Convert 1D array to a 2D column vector

        if len(y.shape) == 1:
            y = y.reshape(-1, 1)  # Convert 1D array to a 2D column vector

        if X.shape[ 0 ] != y.shape[ 0 ]:
            raise ValueError("Number of rows in X and y must be equal.")
