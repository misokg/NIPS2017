from moe.optimal_learning.python.data_containers import HistoricalData
import numpy as np

__author__ = 'matthiaspoloczek'

def sample_initial_data(problem, num_initial_pts_per_IS):
    points = problem.obj_func_min.get_moe_domain().generate_uniform_random_points_in_domain(num_initial_pts_per_IS)
    points_dict = {}
    vals_dict = {}
    noise_dict = {}
    new_historical_data = HistoricalData(dim=problem.obj_func_min.getDim() + 1)  # increased by one for index of IS
    for IS in problem.obj_func_min.getList_IS_to_query():
        points_dict[IS] = np.hstack((IS * np.ones(num_initial_pts_per_IS).reshape((-1, 1)), points))
        vals_dict[IS] = np.array([-1.0 * problem.obj_func_min.evaluate(IS, pt) for pt in points])
        noise_dict[IS] = np.ones(len(points)) * problem.obj_func_min.noise_and_cost_func(IS, None)[0]
        # note: misoKG will learn the noise from sampled data
        new_historical_data.append_historical_data(points_dict[IS], vals_dict[IS], noise_dict[IS])
    return new_historical_data


def select_startpts_BFGS(list_sampled_points, point_to_start_from, num_multistart, problem):
    '''
    create starting points for BFGS, first select points from previously sampled points,
    but not more than half of the starting points
    :return: numpy array with starting points for BFGS
    '''
    #
    if len(list_sampled_points) > 0:
        indices_chosen = np.random.choice(len(list_sampled_points), int(min(len(list_sampled_points), num_multistart/2. - 1.)), replace=False)
        start_pts = np.array(list_sampled_points)[indices_chosen]
        start_pts = np.vstack((point_to_start_from, start_pts)) # add the point that will be sampled next
    else:
        start_pts = [point_to_start_from]
    # fill up with points from an LHS
    start_pts = np.vstack( (start_pts, problem.obj_func_min.get_moe_domain().generate_uniform_random_points_in_domain(num_multistart-len(start_pts))) )
    return start_pts


def process_parallel_results(parallel_results):
    inner_min = np.inf
    for result in parallel_results:
        if inner_min > result[1]:
            inner_min = result[1]
            inner_min_point = result[0]
    return inner_min, inner_min_point