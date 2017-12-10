from abc import ABCMeta, abstractproperty

from misoRosenbrock.rosenbrock import RosenbrockNew, RosenbrockRemi

class MisoRosenbrockBenchmark(object):
    """ Base class for all miso atoext benchmark problems.
    """
    __metaclass__ = ABCMeta

    def __init__(self, replication_no, method_name, obj_func_idx, bucket):
        self.replication_no = replication_no
        self.method_name = method_name
        self._hist_data = None
        self._bucket = bucket
        self._obj_func = [RosenbrockRemi(mult=1.0), RosenbrockNew(mult=1.0)]
        self._obj_func_idx = obj_func_idx

    @abstractproperty
    def num_is_in(self):
        pass

    @property
    def obj_func_min(self):
        return self._obj_func[self._obj_func_idx]

    @property
    def num_iterations(self):
        return 15
        # increase for more steps of the BO algorithm

    @property
    def truth_is(self):
        return 0

    @property
    def exploitation_is(self):
        return 1

    @property
    def list_sample_is(self):
        return range(2)

    @property
    def hist_data(self):
        return self._hist_data

    @hist_data.setter
    def set_hist_data(self, data):
        self._hist_data = data

class MisoRosenbrockBenchmarkMkg(MisoRosenbrockBenchmark):

    def __init__(self, replication_no, obj_func_idx, bucket):
        super(MisoRosenbrockBenchmarkMkg, self).__init__(replication_no, "mkg", obj_func_idx, bucket)
        self._hist_data = None

    @property
    def num_is_in(self):
        return 1    # This should be idx of the last IS, in this case, is 2

class_collection = {
    "miso_rb_benchmark_mkg": MisoRosenbrockBenchmarkMkg,
}
