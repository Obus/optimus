import abc

from .base import Parametrizable


class LineSearch(Parametrizable):

    @abc.abstractmethod
    def __call__(self, func, grad, x, d):
        raise NotImplementedError()


class WolfeLS(LineSearch):

    def __call__(self, func, grad, x, d):
        return sp.optimize.line_search(func, grad, x, d)[0]

    def params(self):
        return {}


class ScalarLS(LineSearch):

    def __init__(self, method='brent'):
        self.method = method

    def __call__(self, func, grad, x, d):
        return scipy.optimize.minimize_scalar(
            lambda a: func(x + a * d), method=self.method, tol=1e-30
        ).x

    def params(self):
        return {'method': self.method}

