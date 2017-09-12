import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class HopfieldNetwork(BaseEstimator, TransformerMixin):
    '''
    Class for a Hopfield neural network.
    '''

    def __init__(self):
        self.p_ = 0  # The number of patterns
        self.n_ = 0  # The number of neurons
        self.patterns_ = []  # The patterns
        self.fitted_ = False  # has the model been trained?
        self.W_ = np.zeros((0, 0))  # The weights matrix

    def fit(self, X, y=None):
        '''
        Fit the Hopfield network to the list of patterns.

        Params:
        -------

        X: np.matrix
            A `np.matrix` in which the rows represent the patterns to memorize
            and the columns the neurons of the network.
        Y : any
            Ignored. Kept for compatibility reasons with the scikit-learn API.
        '''
        self.patterns_ = X
        self.p_, self.n_ = X.shape

        op_sum = np.sum([np.outer(v, v) for v in X], axis=0)

        # Matrix used to cancel the diagonal of the weights matrix 'W_'
        D = self.p_ * np.identity(self.n_)

        self.W_ = (1/self.n_) * (op_sum - D)
        self.fitted_ = True
        return self

    def patterns_to_neurons_ratio(self):
        return self.p_ / self.n_
