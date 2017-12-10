import numpy

from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.domain import TensorProductDomain

class RosenbrockRemi(object):
    """ test function for entropy search: IS0 is truth IS, IS 1 is biased
    This setting is from Lam's paper
    u = 0, v = 0.1
    """
    def __init__(self, mult=1.0):
        """
        :param mult: control whether the optimization problem is maximizing or minimizing, and default is minimizing
        """
        self._dim = 2
        self._search_domain = numpy.repeat([[-2., 2.]], 2, axis=0)
        self._num_IS = 2
        self._mult = mult
        self._meanval = 456.3

    def getSearchDomain(self):
        return self._search_domain

    def evaluate(self, IS, x, true_obj=False):
        """ Global optimum is 0 at (1, 1)
        :param IS: index of information source, 1, ..., M
        :param x[2]: 2d numpy array
        """
        val = pow(1. - x[0], 2.0) + 100. * pow(x[1] - pow(x[0], 2.0), 2.0) - self._meanval
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
        return 'rbRemi'

    def getDim(self):
        return self._dim

    def getNumIS(self):
        return self._num_IS

    def getList_IS_to_query(self):
        return [0,1]

    def get_moe_domain(self):
        return TensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in self._search_domain])

    def get_meanval(self):
        return self._meanval

class RosenbrockNew(RosenbrockRemi):
    """ test function for entropy search: IS0 is unbiased IS, IS 1 is biased
    u = 1, v = 2
    """
    def __init__(self, mult=1.0):
        RosenbrockRemi.__init__(self, mult)
        self._meanval = 456.3

    def evaluate(self, IS, x, true_obj=False):
        """ Global optimum is 0 at (1, 1)
        :param IS: index of information source, 1, ..., M
        :param x[2]: 2d numpy array
        """
        val = pow(1. - x[0], 2.0) + 100. * pow(x[1] - pow(x[0], 2.0), 2.0) - self._meanval
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
        return 'rbNew'

if __name__ == "__main__":
    rb_remi = RosenbrockRemi()
    python_search_domain = TensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in rb_remi._search_domain])
    pts = python_search_domain.generate_uniform_random_points_in_domain(1000)
    print "rb remi"
    print "IS0: {0}".format(numpy.mean([rb_remi.evaluate(0, x) for x in pts]))
    print "IS1: {0}".format(numpy.mean([rb_remi.evaluate(1, x) for x in pts]))
    print "rb new"
    rb_new = RosenbrockNew()
    print "IS0: {0}".format(numpy.mean([rb_new.evaluate(0, x) for x in pts]))
    print "IS1: {0}".format(numpy.mean([rb_new.evaluate(1, x) for x in pts]))
