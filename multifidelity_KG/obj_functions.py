import cPickle as pickle

import numpy
from moe.optimal_learning.python.data_containers import HistoricalData
from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.domain import TensorProductDomain as pythonTensorProductDomain
from moe.optimal_learning.python.python_version.gaussian_process import GaussianProcess

from multifidelity_KG.model.covariance_function import MixedSquareExponential


class Branin(object):
    def __init__(self):
        self._info_dict = {'dim': 2, 'search_domain': numpy.repeat([[-15., 15.]], 2, axis=0),
                           'hyper_domain': numpy.array([[10., 100.], [0.1, 15.], [0.1, 15.]]), 'num_init_pts': 15}

    @property
    def info(self):
        return self._info_dict

    def evaluate(self, x):
        """ This function is usually evaluated on the square x_1 \in [-5, 10], x_2 \in [0, 15]. Global minimum
        is at x = [-pi, 12.275], [pi, 2.275] and [9.42478, 2.475] with minima f(x*) = 0.397887.

            :param x[2]: 2-dim numpy array
        """
        a = 1
        b = 5.1 / (4 * pow(numpy.pi, 2.0))
        c = 5 / numpy.pi
        r = 6
        s = 10
        t = 1 / (8 * numpy.pi)
        return (a * pow(x[1] - b * pow(x[0], 2.0) + c * x[0] - r, 2.0) + s * (1 - t) * numpy.cos(x[0]) + s)


class Rosenbrock(object):
    def __init__(self, num_IS, mult=1.0):
        """
        :param num_IS: number of information sources
        :param noise_and_cost_func: f(IS, x) returns (noise_var, cost)
        :param mult: control whether the optimization problem is maximizing or minimizing, and default is minimizing
        """
        self._dim = 2
        self._search_domain = numpy.repeat([[-2., 2.]], 2, axis=0)
        self._num_IS = num_IS
        self._mult = mult

    def evaluate(self, IS, x):
        """ Global optimum is 0 at (1, 1)
        :param IS: index of information source, 1, ..., M
        :param x[2]: 2d numpy array
        """
        val = pow(1. - x[0], 2.0) + 100. * pow(x[1] - pow(x[0], 2.0), 2.0) - 10.0   # make the minimum -10.0 for now
        if IS == 1:
            val += 0.
        elif IS == 2:
            val += 0.1 * numpy.sin(10.0 * x[0] + 5.0 * x[1])
        return self._mult * val

    def noise_and_cost_func(self, IS, x):
        if IS == 1:
            return 0.001, 1000
        elif IS == 2:
            return 0.01, 1.
        else:
            raise RuntimeError("illegal IS")

class RosenbrockNew(object):
    def __init__(self, num_IS, mult=1.0):
        """
        :param num_IS: number of information sources
        :param noise_and_cost_func: f(IS, x) returns (noise_var, cost)
        :param mult: control whether the optimization problem is maximizing or minimizing, and default is minimizing
        """
        self._dim = 2
        self._search_domain = numpy.repeat([[-2., 2.]], 2, axis=0)
        self._num_IS = num_IS
        self._mult = mult

    def evaluate(self, IS, x):
        """ Global optimum is 0 at (1, 1)
        :param IS: index of information source, 1, ..., M
        :param x[2]: 2d numpy array
        """
        val = pow(1. - x[0], 2.0) + 100. * pow(x[1] - pow(x[0], 2.0), 2.0) - 10.0   # make the minimum -10.0 for now
        if IS == 0:
            return self._mult * val
        elif IS == 1:
            val += numpy.random.normal()
        elif IS == 2:
            val += 2. * numpy.sin(10.0 * x[0] + 5.0 * x[1])
        return self._mult * val

    def noise_and_cost_func(self, IS, x):
        if IS == 1:
            return 1.0, 50.0
        elif IS == 2:
            return 5.0, 1.0
        else:
            raise RuntimeError("illegal IS")

class RosenbrockNoiseFree(object):
    def __init__(self, num_IS, mult=1.0):
        """
        :param num_IS: number of information sources
        :param noise_and_cost_func: f(IS, x) returns (noise_var, cost)
        :param mult: control whether the optimization problem is maximizing or minimizing, and default is minimizing
        """
        self._dim = 2
        self._search_domain = numpy.repeat([[-2., 2.]], 2, axis=0)
        self._num_IS = num_IS
        self._mult = mult

    def evaluate(self, IS, x):
        """ Global optimum is 0 at (1, 1)
        :param IS: index of information source, 1, ..., M
        :param x[2]: 2d numpy array
        """
        val = pow(1. - x[0], 2.0) + 100. * pow(x[1] - pow(x[0], 2.0), 2.0) - 10.0   # make the minimum -10.0 for now
        if IS == 1:
            val += 0.
        elif IS == 2:
            val += 0.1 * numpy.sin(10.0 * x[0] + 5.0 * x[1])
        return self._mult * val

    def noise_and_cost_func(self, IS, x):
        if IS == 1:
            return 0.001, 1000
        elif IS == 2:
            return 1e-6, 1.
        else:
            raise RuntimeError("illegal IS")

class RosenbrockNewNoiseFree(object):
    def __init__(self, num_IS, mult=1.0):
        """
        :param num_IS: number of information sources
        :param noise_and_cost_func: f(IS, x) returns (noise_var, cost)
        :param mult: control whether the optimization problem is maximizing or minimizing, and default is minimizing
        """
        self._dim = 2
        self._search_domain = numpy.repeat([[-2., 2.]], 2, axis=0)
        self._num_IS = num_IS
        self._mult = mult

    def evaluate(self, IS, x, true_obj=False):
        """ Global optimum is 0 at (1, 1)
        :param IS: index of information source, 1, ..., M
        :param x[2]: 2d numpy array
        """
        val = pow(1. - x[0], 2.0) + 100. * pow(x[1] - pow(x[0], 2.0), 2.0) - 10.0   # make the minimum -10.0 for now
        if true_obj:
            return self._mult * val
        if IS == 0:
            return self._mult * val
        elif IS == 1:
            val += numpy.random.normal()
        elif IS == 2:
            val += 2. * numpy.sin(10.0 * x[0] + 5.0 * x[1])
        return self._mult * val

    def noise_and_cost_func(self, IS, x):
        if IS == 1:
            return 1.0, 50.0
        elif IS == 2:
            return 1e-6, 1.0
        else:
            raise RuntimeError("illegal IS")


class RosenbrockNoiseFreePES(object):
    """ test function for entropy search: IS0 is truth IS, IS 1 is biased
    """
    def __init__(self, mult=1.0):
        """
        :param mult: control whether the optimization problem is maximizing or minimizing, and default is minimizing
        """
        self._dim = 2
        self._search_domain = numpy.repeat([[-2., 2.]], 2, axis=0)
        self._num_IS = 2
        self._mult = mult

    def evaluate(self, IS, x, true_obj=False):
        """ Global optimum is 0 at (1, 1)
        :param IS: index of information source, 1, ..., M
        :param x[2]: 2d numpy array
        """
        val = pow(1. - x[0], 2.0) + 100. * pow(x[1] - pow(x[0], 2.0), 2.0) - 10.0   # make the minimum -10.0 for now
        if true_obj:
            return self._mult * val
        if IS != 0 and IS != 1:
            raise ValueError("IS has to be 0 or 1")
        if IS == 1:
            val += 0.1 * numpy.sin(10.0 * x[0] + 5.0 * x[1])
        return self._mult * val

    def noise_and_cost_func(self, IS, x):
        if IS != 0 and IS != 1:
            raise ValueError("IS has to be 0 or 1")
        return (0.001, 1000) if IS == 0 else (1e-6, 1.)

    def getFuncName(self):
        return 'rbpes'

    def getDim(self):
        return self._dim

    def getNumIS(self):
        return self._num_IS

    def getList_IS_to_query(self):
        return [0,1]

class RosenbrockNewNoiseFreePES(RosenbrockNoiseFreePES):
    """ test function for entropy search: IS0 is unbiased IS, IS 1 is biased
    """
    def __init__(self, mult=1.0):
        RosenbrockNoiseFreePES.__init__(self, mult)

    def evaluate(self, IS, x, true_obj=False):
        """ Global optimum is 0 at (1, 1)
        :param IS: index of information source, 1, ..., M
        :param x[2]: 2d numpy array
        """
        val = pow(1. - x[0], 2.0) + 100. * pow(x[1] - pow(x[0], 2.0), 2.0) - 10.0   # make the minimum -10.0 for now
        if true_obj:
            return self._mult * val
        if IS != 0 and IS != 1:
            raise ValueError("IS has to be 0 or 1")
        elif IS == 0:
            val += numpy.random.normal()
        elif IS == 1:
            val += 2. * numpy.sin(10.0 * x[0] + 5.0 * x[1])
        return self._mult * val

    def noise_and_cost_func(self, IS, x):
        if IS != 0 and IS != 1:
            raise ValueError("IS has to be 0 or 1")
        return (1., 50) if IS == 0 else (1e-6, 1.)

    def getFuncName(self):
        return 'rbnewpes'

class RandomGP(object):
    def __init__(self, num_IS, noise_and_cost_func, dim, hyper_params, historical_data=None):
        # fixed typo in param hyper_params
        self._info_dict = {'dim': dim, 'search_domain': numpy.repeat([[-10., 10.]], dim, axis=0),
                           'hyper_params': hyper_params, 'num_init_pts': dim}
        self._sample_var_1 = 0.01
        self._sample_var_2 = 0.2
        self._cov = MixedSquareExponential(hyper_params, dim, num_IS)
        if historical_data is not None:
            self._gp = GaussianProcess(self._cov, historical_data)

    @property
    def info(self):
        return self._info_dict

    def generate_data(self, num_data):
        python_search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in self._info_dict['search_domain']])
        data = HistoricalData(self._info_dict['dim'])
        init_pts = python_search_domain.generate_uniform_random_points_in_domain(2)
        init_pts[:,0] = numpy.zeros(2)
        data.append_historical_data(init_pts, numpy.zeros(2), numpy.ones(2) * self._sample_var_1)
        gp = GaussianProcess(self._cov, data)
        points = python_search_domain.generate_uniform_random_points_in_domain(num_data)
        for pt in points:
            pt[0] = numpy.ceil(numpy.random.uniform(high=2.0, size=1))
            sample_var = self._sample_var_1 if pt[0] == 1 else self._sample_var_2
            val = gp.sample_point_from_gp(pt, sample_var)
            data.append_sample_points([[pt, val, sample_var], ])
            gp = GaussianProcess(self._cov, data)
        return data

    def evaluate(self, point):
        """
        :param point: the first entry specifies which information source (1, ..., M) to use
        :return:
        """
        sample_var = self._sample_var_1 if point[0] == 1 else self._sample_var_2
        return self._gp.sample_point_from_gp(point, noise_variance=sample_var)


if __name__ == "__main__":
    hyper_params = 5.0 * numpy.ones(4)
    gp = RandomGP(3, hyper_params)
    hist_data = gp.generate_data(1000)
    hist_dict = {'points': hist_data.points_sampled, 'values': hist_data.points_sampled_value,
                 'vars': hist_data.points_sampled_noise_variance, 'dim': 3, 'hyper_params': hyper_params}
    pickle.dump(hist_dict, open('random_gp_3d', 'wb'))
    print numpy.min(hist_data.points_sampled_value)
