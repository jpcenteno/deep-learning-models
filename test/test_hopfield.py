import unittest
import numpy.testing as npt

from deep_learning_models.hopfield import *

class HopfieldNetworkTest(unittest.TestCase):

    def setUp(self):
        '''
        Set up of test fixtures.
        '''
        self.net = HopfieldNetwork()

        # A dummy list of patterns.
        self.patterns_A = [np.array([-1, 1] * 50), np.array([1, 1] * 50)]
        # The same, in matrix form.
        self.patterns_A_matrix = np.matrix(self.patterns_A)

    def test_fit(self):
        '''
        Tests that the `fit` method yields the correct state for some simple
        matrix of pattens.
        '''

        # Test data
        X = np.matrix([[1,  1,  1,  1],
                       [1, -1,  1, -1],
                       [1,  1, -1, -1]])
        W = (1/4) * np.matrix([[ 0,  1,  1, -1],
                               [ 1,  0, -1,  1],
                               [ 1, -1,  0,  1],
                               [-1,  1,  1,  0]])

        self.assertFalse(self.net.fitted_)  # initial value should be False.

        self.net.fit(X)  # Fit the network to the patterns.

        # Correct weight matrix
        npt.assert_array_equal(self.net.W_, W)

        self.assertEqual(self.net.p_, 3)  # Number of patterns
        self.assertEqual(self.net.n_, 4)  # Number of neurons
        npt.assert_array_equal(self.net.patterns_, X)  # Patterns
        self.assertTrue(self.net.fitted_)  # The model has been fitted
        self.assertEqual(self.net.patterns_to_neurons_ratio(), 3/4)

    def test_predict(self):
        pass  # TODO

    def test_pattern_self_convergence(self):
        '''
        Every pattern of a trained Hopfield network should converge to
        itself when passed as input.
        '''
        pass  # TODO

    def test_pattern_self_convergence_inverse(self):
        '''
        Every inverse pattern of a trained Hopfield network should converge to
        itself when passed as input.
        '''
        pass  # TODO

    def test_patterns_to_neurons_ratio(self):
        '''
        Tests the `HopfieldNetwork.patterns_to_neurons_ratio` method.
        '''
        pass  # TODO

    def test_is_pattern_without_inverses(self):
        '''
        The `is_pattern` method returns true if some pattern is a pattern of
        the fitted model. If the network is untrained, it should return False.

        When inverses is set to False, the model will not check that the
        pattern is the inverse of some explicitly defined pattern.
        '''
        pass  # TODO

    def test_is_pattern_with_inverse(self):
        '''
        The `is_pattern` method returns true if some pattern is a pattern of
        the fitted model. If the network is untrained, it should return False.

        When inverses is set to True, the model will check if the
        pattern is the inverse of some explicitly defined pattern.
        '''
        pass  # TODO

