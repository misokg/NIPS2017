
import numpy
import scipy.linalg
import scipy.optimize
from joblib import Parallel, delayed
from operator import itemgetter

from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.domain import TensorProductDomain as pythonTensorProductDomain

__author__ = 'matthiaspoloczek'

'''
Optimization of the Hyperparameters of the MISO statistical model: signal variance and length scales of the SE kernel,
and observational noise parameter.

'''


def covariance(dim, hyperparameters, lengths_sq_0, point_one, point_two):
    '''
    Calculate the covariance between two points (not accounting for noise) under the MISO model

    :param point_one: Note that point_one[0] gives the index of the IS, and point_one[1:] the coordinates
    :param point_two: Note that point_two[0] gives the index of the IS, and point_two[1:] the coordinates
    :return: the covariance between point_one and point_two under the MISO model
    '''
    result = hyperparameters[0] * numpy.exp(
        -0.5 * numpy.divide(numpy.power(point_two[1:] - point_one[1:], 2.0), lengths_sq_0).sum())

    # If both points observed at same IS (but not IS 0), account for model discrepancy
    if numpy.abs(point_one[0] - point_two[0]) < 1e-10 and int(point_one[0]) > 0:
        l = int(point_one[0])
        result += hyperparameters[l * dim] * numpy.exp(
            -0.5 * numpy.divide(numpy.power(point_two[1:] - point_one[1:], 2.0),
                                numpy.power(hyperparameters[(l * dim + 1):(l + 1) * dim], 2.0)).sum())
    # print "point one {0}, point two {1}, result {2}".format(point_one, point_two, result)
    return result



def compute_covariance_matrix(dim, hyperparameters_excl_noise, noise_hyperparameters, points_sampled):
    '''
    Compute the covariance matrix of the points that have been sampled so far

    Args:
        dim: 1 + the dimension of the search space that the points are from
        hyperparameters_excl_noise: the signal variance followed by the length scales, enumerated for IS 0,1,...
        noise_hyperparameters: A list of the hyperparameters of the observational noise for IS 0,1,
        points_sampled: the points that have been sampled so far

    Returns: the covariance matrix

    '''

    lengths_sq_0 = numpy.power(hyperparameters_excl_noise[1:dim], 2.0)
    # the squared length scales of IS 0 (pre-computed since they are used for each pt)

    cov_mat = numpy.zeros((points_sampled.shape[0], points_sampled.shape[0]), order='F')
    for j, point_two in enumerate(points_sampled):
        for i, point_one in enumerate(points_sampled[j:, ...], start=j):
            cov_mat[i, j] = covariance(dim, hyperparameters_excl_noise, lengths_sq_0, point_one, point_two)
            if i != j:
                cov_mat[j, i] = cov_mat[i, j]

        # add noise on the main diagonal
        cov_mat[j, j] += numpy.power(noise_hyperparameters[int(point_two[0])], 2.0)
        # We have to square the noise hyperparameter. This is how it is done in MOE:
        # noise_variance: the ``\sigma_n^2`` (noise variance) associated w/observation, points_sampled_value.
        # In build_covariance_matrix() in moe/optimal_learning/python/python_version/python_utils.py:
        # noise_variance: i-th entry is amt of noise variance to add to i-th diagonal entry; i.e., noise measuring i-th point

    return cov_mat



def compute_log_likelihood(K_chol, K_inv_y, points_sampled_value):
    '''
    Compute the _log_likelihood_type measure.

    .. NOTE:: These comments are copied from LogMarginalLikelihoodEvaluator::ComputeLogLikelihood in gpp_model_selection.cpp.

    ``log p(y | X, \theta) = -\frac{1}{2} * y^T * K^-1 * y - \frac{1}{2} * \log(det(K)) - \frac{n}{2} * \log(2*pi)``
    where n is ``num_sampled``, ``\theta`` are the hyperparameters, and ``\log`` is the natural logarithm.  In the following,
    ``term1 = -\frac{1}{2} * y^T * K^-1 * y``
    ``term2 = -\frac{1}{2} * \log(det(K))``
    ``term3 = -\frac{n}{2} * \log(2*pi)``

    For an SPD matrix ``K = L * L^T``,
    ``det(K) = \Pi_i L_ii^2``
    We could compute this directly and then take a logarithm.  But we also know:
    ``\log(det(K)) = 2 * \sum_i \log(L_ii)``
    The latter method is (currently) preferred for computing ``\log(det(K))`` due to reduced chance for overflow
    and (possibly) better numerical conditioning.

    :param K_chol: The cholesky factor of the covariance matrix of the sampled points.
    :param K_inv_y:
    :param points_sampled_value: the observed values at the points we sampled so far
    :return: value of log_likelihood evaluated at hyperparameters (``LL(y | X, \theta)``)
    :rtype: float64
    '''

    # covariance_matrix = compute_covariance_matrix(dim, hyperparameters, lengths_sq_0, noise_hyperparameters, points_sampled)
    # K_chol = scipy.linalg.cho_factor(covariance_matrix, lower=True, overwrite_a=True)

    log_marginal_term2 = -numpy.log(K_chol[0].diagonal()).sum()

    # K_inv_y = scipy.linalg.cho_solve(K_chol, points_sampled_value)
    log_marginal_term1 = -0.5 * numpy.inner(points_sampled_value, K_inv_y)

    log_marginal_term3 = -0.5 * numpy.float64(points_sampled_value.size) * numpy.log(2.0 * numpy.pi)
    return log_marginal_term1 + log_marginal_term2 + log_marginal_term3



def compute_delta_observations(num_IS, points_sampled, points_sampled_value):
    '''
    Returns a list of lists, where the list[0] contains the observations of IS0, and list[i] contains the differences between
    observations j at IS i and observation j at IS 0.
    It supposes that observations are sorted.

    :param num_IS: number of IS
    :param points_sampled: the points that have been sampled. The first entry of each point gives the IS
    :param points_sampled_value: the corresponding observations
    :return: list as described above
    '''

    if (len(points_sampled) != len(points_sampled_value)) or (len(points_sampled) % num_IS != 0):
        # is the dataset correct, do we have the same number of points for each IS
        return None

    num_pts_per_IS = len(points_sampled) / num_IS
    list_of_diff_values = [points_sampled_value[:num_pts_per_IS]]
    for IS in xrange(1,num_IS):
        diff_values = []
        for i in xrange(num_pts_per_IS):
            for index in xrange(1,len(points_sampled[0])):
                if (abs( points_sampled[i][index] - points_sampled[IS * num_pts_per_IS + i][index]) ) > 1e-3 \
                        or (abs(points_sampled[i][0] - 0.0) > 1e-3) \
                        or (abs(points_sampled[IS * num_pts_per_IS + i][0] - IS) > 1e-3):
                    print 'points do not match: ' + str(points_sampled[i]) + ', ' + str(points_sampled[IS * num_pts_per_IS + i])
                    return None

            diff_values.append(points_sampled_value[IS * num_pts_per_IS + i] - points_sampled_value[i])

        list_of_diff_values.append(numpy.array(diff_values))
    return list_of_diff_values



def compute_hyper_prior(num_IS, problem_search_domain, points_sampled, points_sampled_value):
        prior_mean = []
        # print points_sampled
        # print points_sampled_value
        list_diff_pts = compute_delta_observations(num_IS, points_sampled, points_sampled_value)
        for IS in xrange(num_IS):
            mean = numpy.concatenate(([max(0.01, numpy.var(list_diff_pts[IS]))],
                                     [(d[1]-d[0]) for d in problem_search_domain]))
            # print mean
            prior_mean.extend( mean ) # mean for signal variances of the GPs of the IS

        prior_mean = numpy.array(prior_mean)
        prior_sig = numpy.diag(numpy.power(prior_mean/2., 2.0)) # variance
        # print prior_mean
        # print prior_sig
        hyper_prior = NormalPrior(prior_mean, prior_sig)

        return hyper_prior



class NormalPrior(object):
    '''
    Jialei's class for a multivariate normal prior
    '''

    def __init__(self, mu, sig):
        self._mu = mu
        self._sig_inv = numpy.linalg.inv(sig)

    def compute_log_likelihood(self, x):
        x_mu = (x - self._mu).reshape((-1, 1))
        return -0.5 * numpy.dot(x_mu.T, numpy.dot(self._sig_inv, x_mu))

    def compute_grad_log_likelihood(self, x):
        x_mu = (x - self._mu).reshape((-1, 1))
        return -0.5 * numpy.dot(self._sig_inv + self._sig_inv.T, x_mu).flatten()



def hyper_opt(num_IS, dim, points_sampled, points_sampled_value, init_hyper, hyper_bounds, approx_grad, hyper_prior=None):
    '''
    Hyperparameter optimization
    Args:
        num_IS: total number of information sources (IS), including truthIS as IS 0
        dim: 1 + the dimension of the search space that the points are from
        points_sampled:
        points_sampled_value:
        init_hyper: starting point of hyperparameters
        hyper_bounds: list of (lower_bound, upper_bound)
        approx_grad: bool
        hyper_prior:

    Returns: (optimial hyper, optimal marginal loglikelihood, function output)

    '''

    # hyper_bounds = [(0.0,numpy.inf) for hyperparameter in init_hyper]

    def obj_func(x):
        '''
        The negative marginal loglikelihood for hyperparameters x
        Args:
            x: the hyperparameters, including noise hyperparameters appended to the hyperparameters of the kernels

        Returns: The negated value of the marginal loglikelihood at hyperparameters x
        '''

        # split x into hyperparameters and noise_hyperparameters
        # For each IS there are dim signal variances and length scales
        hyperparameters_without_noise = x[:(num_IS * dim)]
        noise_hyperparameters = x[(num_IS * dim):]
        # print 'hyperparameters_without_noise = ' + str(hyperparameters_without_noise)
        # print 'noise_hyperparameters = ' + str(noise_hyperparameters)

        # # compute the parts of the marginal loglikelihood
        covariance_matrix = compute_covariance_matrix(dim, hyperparameters_without_noise, noise_hyperparameters, points_sampled)

        K_chol = scipy.linalg.cho_factor(covariance_matrix, lower=True, overwrite_a=True)
        K_inv_y = scipy.linalg.cho_solve(K_chol, points_sampled_value)

        if hyper_prior is not None:
            # This BFGS minimizes but we wish to maximize, thus negate the log marginal likelihood + log prior
            return -1.0 * ( compute_log_likelihood(K_chol, K_inv_y, points_sampled_value)
                           + hyper_prior.compute_log_likelihood(hyperparameters_without_noise) )
            #
            # # impose hyper_prior on signal variances
            # mu_x = []
            # for IS in xrange(num_IS):
            #     mu_x.append(x[IS]) # leverage that signal variance of IS i is at position i of hyperparameters
            # mu_x = numpy.array(mu_x)
            # return -1.0 * ( compute_log_likelihood(K_chol, K_inv_y, points_sampled_value)
            #                + hyper_prior.compute_log_likelihood(mu_x) )

        else:
            # This BFGS minimizes but we wish to maximize, thus negate the log marginal likelihood
            return -1.0 * compute_log_likelihood(K_chol, K_inv_y, points_sampled_value)

        ### with exception handling
        # compute the parts of the marginal loglikelihood
        # try:
        #     K_chol = scipy.linalg.cho_factor(covariance_matrix, lower=True, overwrite_a=True)
        #     K_inv_y = scipy.linalg.cho_solve(K_chol, points_sampled_value)
        # except scipy.linalg.LinAlgError as e:
        #     if 'leading minor not positive definite' in str(e):
        #         # your error handling block
        #         print 'Caught exception:\n' + str(e)
        #         print 'Returning smallest possible marginal likelihood'
        #         return numpy.finfo('d').max # output the worst possible value
        #     else:
        #         raise scipy.linalg.LinAlgError
        # # we minimize, thus negate the log marginal likelihood
        # return -1.0 * compute_log_likelihood(K_chol, K_inv_y, points_sampled_value)

    # disp = 1 enables output
    return scipy.optimize.fmin_l_bfgs_b(func=obj_func, x0=init_hyper, args=(), approx_grad=True,
                                        bounds=hyper_bounds, m=10, factr=10.0, pgtol=0.01,
                                        epsilon=1e-08, iprint=-1, maxfun=15000, maxiter=100, disp=0, callback=None)

    ### TODO Implement the gradient of the marginal log likelihood
    # def grad_func(x):
    #     gp_likelihood.set_hyperparameters(x)
    #
    #     covariance_matrix = compute_covariance_matrix(dim, hyperparameters, lengths_sq_0, noise_hyperparameters, points_sampled)
    #     K_chol = scipy.linalg.cho_factor(covariance_matrix, lower=True, overwrite_a=True)
    #     K_inv_y = scipy.linalg.cho_solve(K_chol, points_sampled_value)
    #
    #     return -1.0 * compute_grad_log_likelihood(K_chol, K_inv_y, points_sampled_value)
    #
    # return scipy.optimize.fmin_l_bfgs_b(func=obj_func, x0=init_hyper, fprime=grad_func, args=(), approx_grad=approx_grad,
    #                                     bounds=hyper_bounds, m=10, factr=10.0, pgtol=0.01,
    #                                     epsilon=1e-08, iprint=-1, maxfun=15000, maxiter=100, disp=1, callback=None)



def generate_hyperbounds(num_IS, problem_search_domain, upper_bound_noise_variances, upper_bound_signal_variances=1000.):
    '''
    Bounds on each hyperparameter to speed up BFGS. Zero is a trivial lower bound. For the upper bound keep in mind that
    the parameters will be squared, hence in particular the bounds for variances should cover all reasonable values.
    Args:
        num_IS: number of IS, including truth as IS 0
        problem_search_domain: the search domain of the problem, usually problem.obj_func_min._search_domain
        upper_bound_signal_variances: an upper bound on the signal variance and the sample/noise variance

    Returns: bounds as an array of pairs (lower_bound, upper_bound)

    '''
    general_lower_bound = 1e-1 # set a general lower bound for hyperparameters -- allowing 0. seems to cause singularities
    # SPEARMINT uses a tophat prior with lower bound 0.01 on the squared hyperparameters for length scales
    noise_var_lower_bound = 1e-2

    # arbitrary large value for signal variance
    bounds_for_one_IS = [(general_lower_bound, upper_bound_signal_variances)]
    # trivial upper bounds on length scales. Remember that they will be squared.
    bounds_for_one_IS = numpy.append(bounds_for_one_IS, [ (general_lower_bound, (ub-lb)) for (lb,ub) in problem_search_domain])
    # repeat for all IS
    hyper_bounds = numpy.tile(bounds_for_one_IS, num_IS)

    # add bounds on noise/sample variances
    hyper_bounds = numpy.append(hyper_bounds, numpy.tile([ [noise_var_lower_bound, upper_bound_noise_variances] ], num_IS))

    return hyper_bounds.reshape(-1,2) # reshape so that the array becomes an array of pairs (lower_bound, upper_bound)



def optimize_hyperparameters(num_IS, problem_search_domain, points_sampled, points_sampled_value,
                             upper_bound_noise_variances = 10., consider_small_variances = True,
                             hyper_prior = None, num_restarts = 32, num_jobs = 16):
    '''
    Fit hyperparameters from data using MLE or MAP (described in Poloczek, Wang, and Frazier 2016)

    :param num_IS: The total number of information sources
    :param problem_search_domain: The search domain of the benchmark, as provided by the benchmark
    :param points_sampled: An array that gives the points sampled so far. Each points has the form [IS dim0 dim1 ... dimn]
    :param points_sampled_value: An array that gives the values observed at the points in same ordering
    :param upper_bound_noise_variances: An upper bound on the search interval for the noise variance parameters (before squaring)
    :param consider_small_variances: If true, half of the BFGS starting points have entries for the noise parameters set to a small value
    :param hyper_prior: use prior for MAP estimate if supplied, and do MLE otherwise
    :param num_restarts: number of starting points for BFGS to find MLE/MAP
    :param num_jobs: number of parallelized BFGS instances
    :return: An array with the best found values for the hyperparameters
    '''

    approx_grad = True
    upper_bound_signal_variances = numpy.maximum(10., numpy.var(points_sampled_value)) # pick huge upper bounds

    hyper_bounds = generate_hyperbounds(num_IS, problem_search_domain, upper_bound_noise_variances, upper_bound_signal_variances)
    hyperparam_search_domain = pythonTensorProductDomain([ClosedInterval(bd[0], bd[1]) for bd in hyper_bounds])
    hyper_multistart_pts = hyperparam_search_domain.generate_uniform_random_points_in_domain(num_restarts)
    dim = len(problem_search_domain) +1 #  1 + the dimension of the search space that the points are from

    # best_f = numpy.inf
    for i in xrange(num_restarts):
        init_hyper = hyper_multistart_pts[i]

        # if optimization is enabled, make sure that small variances are checked despite multi-modality
        # this optimization seems softer than using a MAP estimate
        if consider_small_variances and (i % 2 == 0):
            for j in xrange(num_IS):
                init_hyper[-1-j] = 0.1 # use a small value as starting point for noise parameters in BFGS

        # # print init_hyper
        # # print hyper_bounds
        # # print len(init_hyper)
        # # print len(hyper_bounds)
        # # print hyper_multistart_pts.shape
        # # exit(0)
        # # If hypers are optimized sequentially
        # hyper, f, output = hyper_opt(num_IS, dim, points_sampled, points_sampled_value, init_hyper, hyper_bounds, approx_grad)
        # # print output
        # if f < best_f: # recall that we negated the log marginal likelihood when passing it to BFGS
        #     best_hyper = hyper
        #     best_f = f
        # # print "itr {0}, hyper: {1}, negative log marginal likelihood: {2}".format(i, hyper, f)
        #
        # only if opt. hypers in parallel:
        hyper_multistart_pts[i] = init_hyper

    parallel_results = Parallel(n_jobs=num_jobs)(delayed(hyper_opt)(num_IS, dim, points_sampled, points_sampled_value, init_hyper, hyper_bounds, approx_grad, hyper_prior) for init_hyper in hyper_multistart_pts)
    # print min(parallel_results,key=itemgetter(1))
    best_hyper = min(parallel_results,key=itemgetter(1))[0] # recall that we negated the log marginal likelihood when passing it to BFGS
    # print 'best_hyper = ' + str(best_hyper) + ' with -log(prob[Y|D]) = ' \
    #       + str(min(parallel_results,key=itemgetter(1))[1]) \
    #       + ' for upper_bound_noise_variances = ' + str(upper_bound_noise_variances)
    # # hyperparameters_without_noise = best_hyper[:(num_IS * dim)]
    # # noise_hyperparameters = best_hyper[(num_IS * dim):]
    # # print compute_covariance_matrix(dim, hyperparameters_without_noise, noise_hyperparameters, points_sampled)

   #  # test using hypers from big dataset
   #  def obj_func(x):
   #      '''
   #      The negative marginal loglikelihood for hyperparameters x
   #      Args:
   #          x: the hyperparameters, including noise hyperparameters appended to the hyperparameters of the kernels
   #
   #      Returns: The negated value of the marginal loglikelihood at hyperparameters x
   #      '''
   #
   #      # split x into hyperparameters and noise_hyperparameters
   #      # For each IS there are dim signal variances and length scales
   #      hyperparameters_without_noise = x[:(num_IS * dim)]
   #      noise_hyperparameters = x[(num_IS * dim):]
   #      # print 'hyperparameters_without_noise = ' + str(hyperparameters_without_noise)
   #      # print 'noise_hyperparameters = ' + str(noise_hyperparameters)
   #
   #      # # compute the parts of the marginal loglikelihood
   #      covariance_matrix = compute_covariance_matrix(dim, hyperparameters_without_noise, noise_hyperparameters, points_sampled)
   #
   #      K_chol = scipy.linalg.cho_factor(covariance_matrix, lower=True, overwrite_a=True)
   #      K_inv_y = scipy.linalg.cho_solve(K_chol, points_sampled_value)
   #
   #      # This BFGS minimizes but we wish to maximize, thus negate the log marginal likelihood
   #      return -1.0 * compute_log_likelihood(K_chol, K_inv_y, points_sampled_value)
   #  # hypers for RbRemi on large dataset with known noise
   #  init_hyper = numpy.array([6.99174646e+05, 7.26756985e-01, 3.04331525, 1.20070203,
   #                                           1.65571854e-01, 3.28218161e-01, 1e-1, 1e-1])
   # #  # hypers for RbNew on large dataset with known noise
   # #  init_hyper = numpy.array([  6.89212443e+05,   7.06559876e-01,   2.98432914e+00,   2.05984746e+00,
   # # 1.16904675e-01,   2.23726117e-01, 1.0, 1e-1])
   #  print 'hypers from large dataset with known noise have -log p(Y|D) of ' + str(obj_func(init_hyper))
   #  hyper, f, output = hyper_opt(num_IS, dim, points_sampled, points_sampled_value, init_hyper, hyper_bounds, approx_grad)
   #  print 'starting from opt hyper = ' + str(hyper) + ", f=" + str(f) #+ ", output = " + str(output)
   #  exit(0)

    return best_hyper



def create_array_points_sampled_noise_variance(points_sampled, noise_hyperparameters):
    points_sampled_noise_variance = []
    for point in points_sampled:
        # point[0] gives the IS that point[1:] was observed at
        points_sampled_noise_variance.append(noise_hyperparameters[ int(round(point[0],0)) ])

    return numpy.array(points_sampled_noise_variance)



