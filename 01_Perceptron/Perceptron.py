import numpy as np


class Perceptron:

    """Perceptron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting. w[0] = threshold
    errors_ : list
        Number of miss classifications in each epoch.

    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        self.w_ = None  # defined in method fit
        self.errors_ = None

    def fit(self, X, y):

        """Fit training dat.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        """
        self.w_ = np.zeros(1 + X.shape[1])  # First position corresponds to threshold
        self.errors_ = []

        # TODO: Put your code (fit algorithm)
        for i in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):  # A cada iteració aconseguim una mostra i la seva classe
                update = self.eta * (target - self.predict(xi))  # Si la prediccio es correcta update = 0
                self.w_[1:] += update * xi  # actualitzacio dels pesos
                self.w_[0] += update  # actualitzacio del bias
                # Feina extra: calculam els errors de classificacio a cada iteració
                errors += int(update != 0.0)
            self.errors_.append(errors)


    def __net_output(self, X):
        """Calculate net output"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label.
            First calculate the output: (X * weights) + threshold
            Second apply the step function
            Return a list with classes
        """
        # TODO: Put your code
        return np.where(self.__net_output(X) >= 0.0, 1, -1)
