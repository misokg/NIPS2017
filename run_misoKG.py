import numpy as np
from pickle import dump
from joblib import Parallel, delayed
import sys
from operator import itemgetter

from multifidelity_KG.misokg_utils import sample_initial_data, process_parallel_results, select_startpts_BFGS
from multifidelity_KG.model.hyperparameter_optimization_with_noise import optimize_hyperparameters, \
    create_array_points_sampled_noise_variance, compute_hyper_prior
from multifidelity_KG.voi.optimization import *
from problems.identifier import identify_problem

from moe.optimal_learning.python.cpp_wrappers.covariance import MixedSquareExponential as cppMixedSquareExponential
from moe.optimal_learning.python.cpp_wrappers.gaussian_process import GaussianProcessNew
from moe.optimal_learning.python.data_containers import SamplePoint, HistoricalData

'''
The misoKG algorithm.

See the description in the paper "Multi-Information Source Optimization" by 
Matthias Poloczek, Jialei Wang, Peter I. Frazier. Multi-Information Source Optimization, NIPS 2017.

Hyperparameters are learned using the MAP estimate described in the appendix of the paper.
'''

__author__ = 'jialeiwang'
__author__ = 'matthiaspoloczek'

# Read benchmark from argv
argv = sys.argv[1:]
if argv[0].find("kg") < 0:
    raise ValueError("benchmark is not mkg/kg!")
problem = identify_problem(argv, None)
filename_result = 'misoKG_results_repl_' + str(problem.replication_no) +'.pickle'
print 'Results are written into file ' + filename_result

### Fast demo mode
num_x_prime = 2000
num_multistart = 8 # number of starting points when searching for optimum of posterior mean and maximum KG factor
num_threads = -1 # how many jobs to use in parallelization? This uses all CPUs, reduce for parallel runs of misoKG
num_parallel_inst = 1 # how many instances of the benchmark can be run in parallel?
num_initial_pts_per_IS = 10 # how many initial points for each IS
#
### Experiment mode
# num_x_prime = 3000
# num_multistart = 32 # number of starting points when searching for optimum of posterior mean and maximum KG factor
# num_threads = -1 # how many jobs to use in parallelization? This uses all CPUs, reduce for parallel runs of misoKG
# num_parallel_inst = -1 # how many instances of the benchmark can be run in parallel?
# num_initial_pts_per_IS = 5 # how many initial points for each IS

### sample initial points, extend points by IS index, and store them as historical data
problem.set_hist_data = sample_initial_data(problem, num_initial_pts_per_IS)

### mkg begins
kg_gp_cpp = None
num_discretization_before_ranking = num_x_prime * 2
# if the KG/unit_cost drops below this, then sample at optimum of posterior mean.
# The exploitation IS is defined in problem object.
exploitation_threshold = 1e-5
# data containers for pickle storage
list_best = []
list_cost = [] # cumulative cost so far
list_sampled_IS = [] # list of all IS queried in chronol. order
list_sampled_points = [] # list of all points sampled in chron. order
list_sampled_vals = [] # list of corresponding obs
list_noise_var = [] # list of noise variance of observations in chron. order
list_mu_star_truth = [] # list of values at resp. mu_star_points under IS0
list_pending_mu_star_points = []
list_raw_voi = []
init_best_idx = numpy.argmax(problem.hist_data._points_sampled_value[problem.hist_data._points_sampled[:, 0] == problem.truth_is])
best_sampled_val = -1.0 * problem.hist_data._points_sampled_value[init_best_idx]
# minus sign is because vals in hist_data were obtained from obj_func_max, while all values to store are
# from obj_func_min, for consistency
truth_at_init_best_sampled = best_sampled_val
truth_at_best_sampled = truth_at_init_best_sampled
best_mu_star_truth = np.inf
total_cost = 0.0
num_IS = len(problem.obj_func_min.getList_IS_to_query())

# Next iteration of misoKG
# after how many iterations shall we re-optimize the hyperparameters?
hyper_learning_interval = int(numpy.maximum(problem.num_iterations / 5, len(problem.hist_data._points_sampled_value)))

# The hyper prior is computed from the initial dataset, since it requires that all IS are evaluated at the same points
hyper_prior = compute_hyper_prior(num_IS, problem.obj_func_min.getSearchDomain(), problem.hist_data.points_sampled,
                                  problem.hist_data.points_sampled_value)

for kg_iteration in xrange(problem.num_iterations):

    ## Update hyper every hyper_learning_interval-many samples
    if kg_iteration == 0 or kg_gp_cpp.num_sampled % hyper_learning_interval == 0:
        current_hist_data = kg_gp_cpp.get_historical_data_copy() if kg_gp_cpp else problem.hist_data

        num_IS = len(problem.obj_func_min.getList_IS_to_query())
        best_hyper = optimize_hyperparameters(num_IS, problem.obj_func_min.getSearchDomain(),
                                              current_hist_data.points_sampled, current_hist_data.points_sampled_value,
                                              upper_bound_noise_variances=10., consider_small_variances = True,
                                              hyper_prior=hyper_prior,
                                              num_restarts = 16, num_jobs = num_threads)
        # update hyperparameters for noise for each observed value in historical data
        # current_hist_data.points_sampled_noise_variance gives the array with noise values

        # separate hypers for GP and for observational noise
        print "misoKG: repl {0}, itr {1}, best hyper: {2}".format(problem.replication_no, kg_iteration, best_hyper)
        ### Format: IS 0: signal variance and length scales, IS 1: signal variance and length scales, etc.
        ###  Then observational noise for IS 0, IS 1 etc.

        hyperparameters_noise = numpy.power(best_hyper[-num_IS:], 2.0)
        hypers_GP = best_hyper[:-num_IS]

        # update noise in historical data
        updated_points_sampled_noise_variance = create_array_points_sampled_noise_variance(current_hist_data.points_sampled,
                                                                                           hyperparameters_noise)

        # create new Historical data object with updated values
        new_historical_data = HistoricalData(dim=problem.obj_func_min.getDim()+1) # increased by one for index of IS
        new_historical_data.append_historical_data(current_hist_data.points_sampled,
                                                   current_hist_data.points_sampled_value,
                                                   updated_points_sampled_noise_variance)

        # Use new hyperparameters -- this requires instantiating a new GP object
        kg_cov_cpp = cppMixedSquareExponential(hyperparameters=hypers_GP)
        kg_gp_cpp = GaussianProcessNew(kg_cov_cpp, new_historical_data, num_IS_in=problem.num_is_in)
        # kg_cov_cpp is not used afterwards



    ### Find IS and point that maximize KG/cost
    discretization_points = problem.obj_func_min.get_moe_domain().generate_uniform_random_points_in_domain(num_discretization_before_ranking)
    discretization_points = np.hstack((np.zeros((num_discretization_before_ranking,1)), discretization_points))
    all_mu = kg_gp_cpp.compute_mean_of_points(discretization_points)
    sorted_idx = np.argsort(all_mu)
    all_zero_x_prime = discretization_points[sorted_idx[-num_x_prime:], :]
    ### idea ends

    def min_kg_unit(start_pt, IS):
        func_to_min, grad_func = negative_kg_and_grad_given_x_prime(IS, all_zero_x_prime,
                                                                    problem.obj_func_min.noise_and_cost_func, kg_gp_cpp)
        return bfgs_optimization(start_pt, func_to_min, problem.obj_func_min._search_domain) # approximate gradient

    def compute_kg_unit(x, IS):
        return compute_kg_given_x_prime(IS, x, all_zero_x_prime, problem.obj_func_min.noise_and_cost_func(IS, x)[0],
                                        problem.obj_func_min.noise_and_cost_func(IS, x)[1], kg_gp_cpp)

    # For every IS compute a point of maximum KG/cost (here: minimum -1.0 * KG/cost)
    min_negative_kg = np.inf
    list_raw_kg_this_itr = []
    with Parallel(n_jobs=num_threads) as parallel:
        for IS in problem.list_sample_is:

            test_pts = problem.obj_func_min.get_moe_domain().generate_uniform_random_points_in_domain(1000)
            kg_test_pts = parallel(delayed(compute_kg_unit)(pt, IS) for pt in test_pts)

            kg_candidate = test_pts[np.argmax(kg_test_pts)]
            kg_val_at_candidate = np.max(kg_test_pts)
            start_pts = select_startpts_BFGS(list_sampled_points, kg_candidate, num_multistart, problem)
            parallel_results = parallel(delayed(min_kg_unit)(pt, IS) for pt in start_pts)

            # add candidate point to list, remember to negate its KG/cost value since we are looking for the minimum
            parallel_results = np.concatenate((parallel_results, [[kg_candidate, -1.0 * kg_val_at_candidate]]), axis=0)
            inner_min, inner_min_point = process_parallel_results(parallel_results)

            list_raw_kg_this_itr.append(-inner_min * problem.obj_func_min.noise_and_cost_func(IS, inner_min_point)[1])
            if inner_min < min_negative_kg:
                min_negative_kg = inner_min
                point_to_sample = inner_min_point
                sample_is = IS

    # Is the KG (normalized to unit cost) below the threshold? Then exploit by sampling the point of maximum posterior mean
    if -min_negative_kg * problem.obj_func_min.noise_and_cost_func(sample_is, point_to_sample)[1] < exploitation_threshold:
        print "KG search failed, do exploitation"
        mu_star_point = search_mu_star_point(kg_gp_cpp, list_sampled_points, point_to_sample, num_multistart, num_threads, problem)
        point_to_sample = mu_star_point
        sample_is = problem.exploitation_is



    ### make next observation and update GP
    def parallel_func(IS, pt):
        return problem.obj_func_min.evaluate(IS, pt)

    if sample_is == problem.truth_is:
        # we are running the expensive truthIS. Use the time to perform other expensive queries in parallel:
        list_pending_mu_star_points.append(point_to_sample)
        with Parallel(n_jobs=num_parallel_inst) as parallel:
            vals_pending_mu_star_points = parallel(delayed(parallel_func)(problem.truth_is, pt) for pt in list_pending_mu_star_points)

        sample_val = vals_pending_mu_star_points[-1] # the last value belongs to this iteration

        # remove point and obs of this iteration from lists
        list_pending_mu_star_points = list_pending_mu_star_points[:-1]
        vals_pending_mu_star_points = vals_pending_mu_star_points[:-1]

        # add evaluations of mu_star to our list
        list_mu_star_truth.extend(vals_pending_mu_star_points)
        list_pending_mu_star_points = []
    else:
        # just do the cheap observation and defer the expensive one
        sample_val = problem.obj_func_min.evaluate(sample_is, point_to_sample)

    # add point and observation to GP
    # NOTE: while we work everywhere with the values of the minimization problem in the computation, we used the maximization obj values for the GP.
    # That is why here sample_val is multiplied by -1.0
    kg_gp_cpp.add_sampled_points([SamplePoint(np.concatenate(([sample_is], point_to_sample)), -1.0 * sample_val, problem.obj_func_min.noise_and_cost_func(sample_is, point_to_sample)[0])])



    ### Recommendation: Search for point of optimal posterior mean for truth IS
    def find_mu_star(start_pt):
        '''
        Find the optimum of the posterior mean. This is the point that misoKG will recommend in this iteration.
        :param start_pt: starting point for BFGS
        :return: recommended point
        '''
        return bfgs_optimization(start_pt, negative_mu_kg(kg_gp_cpp), problem.obj_func_min._search_domain)

    def search_mu_star_point(kg_gp_cpp, list_sampled_points, point_to_sample, num_multistart, num_threads, problem):
        '''
        Search for point of optimal posterior mean for truth IS
        :param kg_gp_cpp: The GP object
        :param list_sampled_points: points sampled so far
        :param point_to_sample: next point to sample
        :param num_multistart: number of points to start from
        :param num_threads: number of threads to be used
        :param problem: The benchmark object
        :return: point of optimal posterior mean for truth IS (without coordinate that indicates the IS)
        '''
        if len(list_sampled_points) == 0:
            test_pts = numpy.array([point_to_sample])
        else:
            test_pts = np.concatenate(([point_to_sample], numpy.array(list_sampled_points)), axis=0)
        num_random_pts = int(1e4)
        test_pts = np.concatenate((test_pts, problem.obj_func_min.get_moe_domain().generate_uniform_random_points_in_domain(num_random_pts)), axis=0)
        # all points must be extended by the IS 0, i.e. a leading zero must be added to each point
        test_pts_with_IS0 = numpy.insert(test_pts, 0, 0, axis=1)

        # Negate mean values, since KG's GP works with the max version of the problem
        means = -1.0 * kg_gp_cpp.compute_mean_of_points(test_pts_with_IS0)

        mu_star_candidate = test_pts[np.argmin(means)]
        mean_mu_star_candidate = np.min(means)

        start_pts = select_startpts_BFGS(list_sampled_points, mu_star_candidate, num_multistart, problem)
        with Parallel(n_jobs=num_threads) as parallel:
            parallel_results = parallel(delayed(find_mu_star)(pt) for pt in start_pts)

        parallel_results = np.concatenate((parallel_results, [[mu_star_candidate, mean_mu_star_candidate]]), axis=0)
        mu_star_point = min(parallel_results,key=itemgetter(1))[0]
        return mu_star_point

    mu_star_point = search_mu_star_point(kg_gp_cpp, list_sampled_points, point_to_sample, num_multistart, num_threads, problem)
    list_pending_mu_star_points.append(mu_star_point)



    ### perform batched evaluation of mu_star points at truthIS, since they can delay the iteration significantly
    if len(list_pending_mu_star_points) >= num_parallel_inst or (kg_iteration +1 == problem.num_iterations):
        # have we collected several points or is this the last iteration?
        with Parallel(n_jobs=num_parallel_inst) as parallel:
            vals_pending_mu_star_points = parallel(delayed(parallel_func)(problem.truth_is, pt) for pt in list_pending_mu_star_points)

        list_mu_star_truth.extend(vals_pending_mu_star_points)
        list_pending_mu_star_points = []

    # Print progress to stdout
    if len(list_mu_star_truth) > 0:
        print 'repl ' + str(problem.replication_no) + ', it ' + str(kg_iteration) + ', sample IS ' \
              + str(sample_is) + ' at ' + str(point_to_sample) \
              + ', recommendation ' + str(mu_star_point) +' has (observed) value ' + str(list_mu_star_truth[-1]
                                                                              + problem.obj_func_min.get_meanval())
        # ', list_pending_mu_star_points = ' + str(list_pending_mu_star_points)
    else:
        print 'repl ' + str(problem.replication_no) + ', it ' + str(kg_iteration) + ', sample IS ' \
              + str(sample_is) + ' at ' + str(point_to_sample)

    ### Collect data for pickle
    total_cost += problem.obj_func_min.noise_and_cost_func(sample_is, point_to_sample)[1]
    list_cost.append(total_cost)
    list_sampled_IS.append(sample_is)
    list_sampled_points.append(point_to_sample)
    list_sampled_vals.append(sample_val)
    list_noise_var.append(problem.obj_func_min.noise_and_cost_func(sample_is, point_to_sample)[0])
    list_raw_voi.append(list_raw_kg_this_itr)
    result_to_pickle = {
        "cost": np.array(list_cost),
        "sampled_is": np.array(list_sampled_IS),
        "sampled_points": np.array(list_sampled_points),
        "sampled_vals": np.array(list_sampled_vals),
        "mu_star_truth": np.array(list_mu_star_truth),
        "init_best_truth": truth_at_init_best_sampled,
    }
    # write data to pickle at the end of every iteration
    with open(filename_result, "wb") as file:
        dump(result_to_pickle, file)
