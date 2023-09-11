from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from LogisticRegression import LogisticRegression

# Generate a synthetic binary classification dataset with two informative features
X, y = make_classification(
    n_samples=10000,  # Number of samples
    n_features=2,  # Number of features
    n_informative=2,  # Number of informative features
    n_redundant=0,  # Number of redundant features
    n_clusters_per_class=1,  # Number of clusters per class
    random_state=29  # Random seed for reproducibility
)
# splitting X and y into training and testing sets
X_train, X_test, \
    y_train, y_test = train_test_split(X, y,
                                       test_size=0.4,
                                       random_state=1)
# create logistic regression object
reg = LogisticRegression(epochs=100)

# train the model using the training sets
reg.fit(X_train, y_train)
reg.plot2D(X_test, y_test)
