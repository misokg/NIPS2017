import numpy
import scipy.stats

__author__ = 'jialeiwang'

class MultifidelityExpectedImprovement(object):

    def __init__(self, gp_dict, fidelity_var_and_cost_func, list_sample_is):
        """
        :param gp_dict: dict of GPs, key is IS
        :param fidelity_var_and_cost_func: function that computes fidelity_var, cost for (IS, x)
        :param list_sample_is: list of sample IS
        :return:
        """
        self._gp_dict = gp_dict
        self._fidelity_var_and_cost_func = fidelity_var_and_cost_func
        self._list_sample_is = list_sample_is

    def _compute_total_var_list(self, x):
        return numpy.array([self._gp_dict[IS].compute_variance_of_points(x.reshape((1,-1)))[0,0] + self._fidelity_var_and_cost_func(IS, x)[0] for IS in self._list_sample_is])

    def _compute_gp_var_list(self, x):
        return numpy.array([self._gp_dict[IS].compute_variance_of_points(x.reshape((1,-1)))[0,0] for IS in self._list_sample_is])

    def _compute_mu_bar(self, x):
        mu_list = numpy.array([self._gp_dict[IS].compute_mean_of_points(x.reshape((1,-1)))[0] for IS in self._list_sample_is])
        total_var_list = self._compute_total_var_list(x)
        return numpy.divide(mu_list, total_var_list).sum() / numpy.divide(1.0, total_var_list).sum()

    def compute_expected_improvement(self, x):
        mu_bar = self._compute_mu_bar(x)
        var_hat = 1.0 / numpy.divide(1.0, self._compute_gp_var_list(x)).sum()
        y_min = numpy.inf
        for IS in self._list_sample_is:
            y_min = min(y_min, numpy.amin([self._compute_mu_bar(pt) for pt in self._gp_dict[IS]._historical_data.points_sampled]))
        z = (y_min - mu_bar) / numpy.sqrt(var_hat)
        return (y_min - mu_bar) * scipy.stats.norm.cdf(z) + numpy.sqrt(var_hat) * scipy.stats.norm.pdf(z)

    def choose_IS(self, x):
        """
        :param x:
        :return: which IS
        """
        costs = numpy.array([self._fidelity_var_and_cost_func(IS, x)[1] for IS in self._list_sample_is])
        total_var_list = self._compute_total_var_list(x)
        inv_var_bar = total_var_list.sum()
        inv_var_tilt_list = numpy.array([1.0 / self._fidelity_var_and_cost_func(IS, x)[0] + numpy.divide(1.0, total_var_list).sum() - 1.0 / total_var_list[idx] for idx, IS in enumerate(self._list_sample_is)])
        return self._list_sample_is[numpy.argmin(numpy.divide(costs, 1.0 / inv_var_bar - numpy.divide(1.0, inv_var_tilt_list)))]
