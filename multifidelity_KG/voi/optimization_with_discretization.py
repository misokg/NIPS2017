import numpy
import pandas
from joblib import Parallel, delayed

from moe.optimal_learning.python.data_containers import SamplePoint
from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.domain import TensorProductDomain as pythonTensorProductDomain
from moe.optimal_learning.python.python_version.gaussian_process import GaussianProcess
from moe.optimal_learning.python.python_version.covariance import SquareExponential
from moe.optimal_learning.python.python_version.expected_improvement import ExpectedImprovement

from multifidelity_KG.model.covariance_function import MixedSquareExponential
from multifidelity_KG.voi.knowledge_gradient import *

__author__ = 'jialeiwang'

class MultifidelityKGOptimizer(object):
    """ Maximize objective function using multifidelity KG
    """
    def __init__(self, hyperparam, data, num_IS, search_space_dim):
        cov_func = MixedSquareExponential(hyperparameters=hyperparam, total_dim=search_space_dim+1, num_is=num_IS)
        self._gp = GaussianProcess(cov_func, data)
        self._num_IS = num_IS
        self.cumulated_cost = 0

    def _unit_func(self, x, IS, noise_var_x, cost, all_x):
        a, b = compute_a_b(gp=self._gp, x=x, which_IS=IS, noise_var=noise_var_x, all_x=all_x)
        return compute_kg(a, b, cost)

    def _unit_func_test(self, x, IS):
        return x.sum() + IS

    def optimize(self, all_x, noise_vars, costs, num_threads):
        """
        :param all_x: shape (num_all_x, search_space_dim)
        :param noise_vars: numpy array with shape (num_IS, num_all_x)
        :param costs: shape (num_IS, num_all_x)
        :param num_threads:
        :return: (best_IS, best_x, sample_var, cost)
        """
        def unit_func_test(x, IS):
            return x.sum() + IS
        kg_table = numpy.zeros((self._num_IS, len(all_x)))
        with Parallel(n_jobs=num_threads) as parallel:
            for i in range(self._num_IS):
                # parallel_results = parallel(delayed(self._unit_func)(x, i+1, noise_vars[i, n], costs[i, n], all_x) for n, x in enumerate(all_x))
                parallel_results = parallel(delayed(numpy.sqrt)( i+1) for x in (all_x))
                print parallel_results
                kg_table[i, :] = numpy.array(parallel_results)
        best_flatten_idx = numpy.argmax(kg_table)
        best_IS = int(best_flatten_idx / len(all_x)) + 1
        best_x_idx = best_flatten_idx - (best_IS-1) * len(all_x)
        if kg_table[best_IS-1, best_x_idx] != numpy.amax(kg_table):
            raise RuntimeError("index not finding correctly in KG!")
        return best_IS, all_x[best_x_idx, :], noise_vars[best_IS-1, best_x_idx], costs[best_IS-1, best_x_idx]

    def add_sampled_point(self, IS, x, val, noise_var, cost):
        IS_x = numpy.concatenate((numpy.array([IS]), x))
        self._gp.add_sampled_points([SamplePoint(IS_x, val, noise_var)])
        self.cumulated_cost += cost

class EgoOptimizer(object):
    """ Minimize objective function using EGO
    """
    def __init__(self, hyperparam, data, num_IS):
        cov_func = SquareExponential(hyperparameters=hyperparam)
        self._gp = GaussianProcess(cov_func, data)
        self._num_IS = num_IS
        self.cumulated_cost = 0

    def optimize(self, all_x):
        """
        :param all_x:
        :return: (best_point, found_point_successfully)
        """
        ei_evaluator = ExpectedImprovement(self._gp)
        ei_vals = numpy.zeros(len(all_x))
        for i in range(len(ei_vals)):
            ei_evaluator.set_current_point(all_x[i, :].reshape((1,-1)))
            print ei_evaluator.num_to_sample
            ei_vals[i] = ei_evaluator.compute_expected_improvement()
        best_ei = numpy.amax(ei_vals)
        if best_ei > 0:
            return all_x[numpy.argmax(best_ei), :], True
        else:
            return all_x[numpy.random.randint(low=0, high=len(best_ei), size=1), :], False

    def add_sampled_point(self, x, vals, noise_vars, costs):
        """
        :param x: the point to add
        :param vals: sampled values of x using all IS
        :param noise_vars: noise variances at x for all IS
        :param costs: costs at x for all IS
        :return:
        """
        self._gp.add_sampled_points([SamplePoint(x, vals[i], noise_vars[i]) for i in range(self._num_IS)])
        self.cumulated_cost += costs.sum()

