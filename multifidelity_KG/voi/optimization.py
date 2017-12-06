import scipy.optimize

from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.domain import TensorProductDomain as pythonTensorProductDomain
from moe.optimal_learning.python.python_version.expected_improvement import ExpectedImprovement

from multifidelity_KG.voi.knowledge_gradient import *
from multifidelity_KG.voi.multifidelity_expected_improvement import MultifidelityExpectedImprovement
# import sql_util

__author__ = 'jialeiwang'

def bfgs_optimization(start_pt, func_to_minimize, bounds):
    result_x, result_f, output = scipy.optimize.fmin_l_bfgs_b(func=func_to_minimize, x0=start_pt, fprime=None, args=(), approx_grad=True,
                                                              bounds=bounds, m=10, factr=10.0, pgtol=1e-10,
                                                              epsilon=1e-08, iprint=-1, maxfun=15000, maxiter=15000, disp=0, callback=None)
    return result_x, result_f

def bfgs_optimization_grad(start_pt, func_to_min, grad_func, bounds):
    result_x, result_f, output = scipy.optimize.fmin_l_bfgs_b(func=func_to_min, x0=start_pt, fprime=grad_func, args=(), approx_grad=False,
                                                              bounds=bounds, m=10, factr=10.0, pgtol=1e-10,
                                                              epsilon=1e-08, iprint=-1, maxfun=15000, maxiter=15000, disp=0, callback=None)
    return result_x, result_f

def negative_kg_and_grad(IS, search_domain, num_discretization, noise_and_cost_func, gp):
    def negative_kg(x):
        kg, grad_kg = compute_kg_and_grad(IS, x, search_domain, num_discretization, noise_and_cost_func(IS, x)[0], noise_and_cost_func(IS, x)[1], gp)
        return -kg

    def negative_grad_kg(x):
        kg, grad_kg = compute_kg_and_grad(IS, x, search_domain, num_discretization, noise_and_cost_func(IS, x)[0], noise_and_cost_func(IS, x)[1], gp)
        return -grad_kg

    return negative_kg, negative_grad_kg

def negative_kg_and_grad_given_x_prime(IS, all_zero_x_prime, noise_and_cost_func, gp):
    def negative_kg(x):
        return -compute_kg_given_x_prime(IS, x, all_zero_x_prime, noise_and_cost_func(IS, x)[0], noise_and_cost_func(IS, x)[1], gp)

    def negative_grad_kg(x):
        kg, grad_kg = compute_kg_and_grad_given_x_prime(IS, x, all_zero_x_prime, noise_and_cost_func(IS, x)[0], noise_and_cost_func(IS, x)[1], gp)
        return -grad_kg

    return negative_kg, negative_grad_kg

def negative_mu_kg(gp):
    def result(x):
        return -1.0 * gp.compute_mean_of_points(numpy.concatenate(([0], x)).reshape((1,-1)))[0]
    return result

def find_best_mu_kg(gp, domain_bounds, num_multistart):
    search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in domain_bounds])
    start_points = search_domain.generate_uniform_random_points_in_domain(num_multistart)
    min_negative_mu = numpy.inf
    for start_point in start_points:
        x, f = bfgs_optimization(start_point, negative_mu_kg(gp), domain_bounds)
        if min_negative_mu > f:
            min_negative_mu = f
            point = x
    return -min_negative_mu, point

def compute_mu(gp):
    def result(x):
        return gp.compute_mean_of_points(x.reshape((1,-1)))[0]
    return result

def find_best_mu_ei(gp, domain_bounds, num_multistart):
    search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in domain_bounds])
    start_points = search_domain.generate_uniform_random_points_in_domain(num_multistart)
    min_mu = numpy.inf
    for start_point in start_points:
        x, f = bfgs_optimization(start_point, compute_mu(gp), domain_bounds)
        if min_mu > f:
            min_mu = f
            point = x
    return min_mu, point

def optimize_with_multifidelity_ei(gp_list, domain_bounds, num_IS, num_multistart, noise_and_cost_func):
    multifidelity_expected_improvement_evaluator = MultifidelityExpectedImprovement(gp_list, noise_and_cost_func)
    search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in domain_bounds])
    start_points = search_domain.generate_uniform_random_points_in_domain(num_multistart)
    min_negative_ei = numpy.inf

    def negative_ei_func(x):
        return -1.0 * multifidelity_expected_improvement_evaluator.compute_expected_improvement(x)

    for start_point in start_points:
        x, f = bfgs_optimization(start_point, negative_ei_func, domain_bounds)
        if min_negative_ei > f:
            min_negative_ei = f
            point_to_sample = x
    return point_to_sample, multifidelity_expected_improvement_evaluator.choose_IS(point_to_sample), -min_negative_ei

def optimize_with_ego(gp, domain_bounds, num_multistart):
    expected_improvement_evaluator = ExpectedImprovement(gp)
    search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in domain_bounds])
    start_points = search_domain.generate_uniform_random_points_in_domain(num_multistart)
    min_negative_ei = numpy.inf

    def negative_ego_func(x):
        expected_improvement_evaluator.set_current_point(x.reshape((1,-1)))
        return -1.0 * expected_improvement_evaluator.compute_expected_improvement()

    for start_point in start_points:
        x, f = bfgs_optimization(start_point, negative_ego_func, domain_bounds)
        if min_negative_ei > f:
            min_negative_ei = f
            point_to_sample = x
    return point_to_sample, -min_negative_ei
