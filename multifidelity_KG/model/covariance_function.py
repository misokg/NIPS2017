__author__ = 'jialeiwang'

import numpy
from moe.optimal_learning.python.interfaces.covariance_interface import CovarianceInterface

class MixedSquareExponential(CovarianceInterface):

    # covariance_type = SQUARE_EXPONENTIAL_COVARIANCE_TYPE

    def __init__(self, hyperparameters, total_dim, num_is):
        r"""Construct a square exponential covariance object with the specified hyperparameters.

        :param hyperparameters: hyperparameters of the covariance function; index 0 is \alpha (signal variance, \sigma_f^2)
          and index 1..dim are the per-dimension length scales.
        :type hyperparameters: array-like of size dim+1

        """
        self._dim = total_dim         # dimension of IS \times search space
        self._num_is = num_is   # Number of information sources, then including 0th IS (truth), size of hyper should be dim * (num_is+1).
        # Note: it's not (dim+1)*(num_is+1) because dimension of search space is (dim-1), plus the multiplication factor param is dim
        self.set_hyperparameters(hyperparameters)

    @property
    def num_hyperparameters(self):
        """Return the number of hyperparameters of this covariance function."""
        return self._hyperparameters.size

    def get_hyperparameters(self):
        """Get the hyperparameters (array of float64 with shape (num_hyperparameters)) of this covariance."""
        return numpy.copy(self._hyperparameters)

    def set_hyperparameters(self, hyperparameters):
        """Set hyperparameters to the specified hyperparameters; ordering must match."""
        self._hyperparameters = numpy.copy(hyperparameters)
        self._lengths_sq_0 = numpy.power(self._hyperparameters[1:self._dim], 2.0)

    hyperparameters = property(get_hyperparameters, set_hyperparameters)

    def covariance(self, point_one, point_two):
        result = self._hyperparameters[0] * numpy.exp(-0.5 * numpy.divide(numpy.power(point_two[1:] - point_one[1:], 2.0), self._lengths_sq_0).sum())
        if numpy.abs(point_one[0] - point_two[0]) < 1e-10 and int(point_one[0]) > 0:
            l = int(point_one[0])
            result += self._hyperparameters[l*self._dim] * numpy.exp(-0.5 * numpy.divide(numpy.power(point_two[1:] - point_one[1:], 2.0), numpy.power(self._hyperparameters[(l*self._dim+1):(l+1)*self._dim], 2.0)).sum())
        # print "point one {0}, point two {1}, result {2}".format(point_one, point_two, result)
        return result

    def grad_covariance(self, point_one, point_two):
        grad_cov = point_two - point_one
        grad_cov[1:] = grad_cov[1:] / self._lengths_sq_0 * self._hyperparameters[0] * numpy.exp(-0.5 * numpy.divide(numpy.power(point_two[1:] - point_one[1:], 2.0), self._lengths_sq_0).sum())
        if numpy.abs(point_one[0] - point_two[0]) < 1e-10 and int(point_one[0]) > 0:
            l = int(point_one[0])
            grad_cov[1:] += (point_two - point_one)[1:] / numpy.power(self._hyperparameters[(l*self._dim+1):(l+1)*self._dim], 2.0) * self._hyperparameters[l*self._dim] * numpy.exp(-0.5 * numpy.divide(numpy.power(point_two[1:] - point_one[1:], 2.0), numpy.power(self._hyperparameters[(l*self._dim+1):(l+1)*self._dim], 2.0)).sum())
        grad_cov[0] = 0.0   # zeroth component is useless
        return grad_cov

    def hyperparameter_grad_covariance(self, point_one, point_two):
        grad_cov = numpy.zeros(self.num_hyperparameters)
        squared_components = numpy.power(point_two[1:] - point_one[1:], 2.0)
        zeroth_exp_term = numpy.exp(-0.5 * numpy.divide(squared_components, self._lengths_sq_0).sum())
        grad_cov[0] = zeroth_exp_term
        grad_cov[1:self._dim] = numpy.divide(squared_components, numpy.power(self._hyperparameters[1:self._dim], 3.0)) * self._hyperparameters[0] * zeroth_exp_term
        if numpy.abs(point_one[0] - point_two[0]) < 1e-10 and int(point_one[0]) > 0:
            l = int(point_one[0])
            lth_exp_term = numpy.exp(-0.5 * numpy.divide(squared_components, numpy.power(self._hyperparameters[(l*self._dim+1):(l+1)*self._dim], 2.0)).sum())
            grad_cov[l*self._dim] = lth_exp_term
            grad_cov[(l*self._dim+1):(l+1)*self._dim] = numpy.divide(squared_components, numpy.power(self._hyperparameters[(l*self._dim+1):(l+1)*self._dim], 3.0)) * self._hyperparameters[l*self._dim] * lth_exp_term
        return grad_cov

    def hyperparameter_hessian_covariance(self, point_one, point_two):
        r"""Compute the hessian of self.covariance(point_one, point_two) with respect to its hyperparameters.

        TODO(GH-57): Implement Hessians in Python.

        """
        raise NotImplementedError("Python implementation does not support computing the hessian covariance wrt hyperparameters.")
