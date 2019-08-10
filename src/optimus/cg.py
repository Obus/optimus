from .base import Optimizer
from .line_search import ScalarLS


class CG(Optimizer):

    def __init__(
            self,
            line_search=ScalarLS(),
            restart=None
    ):
        self.line_search = line_search
        self.prev_d = None
        self.prev_g = None
        self.restart = restart
        self._step_num = 0

    def step(self, x, func, grad):
        g = grad(x)
        if self.restart is not None and self._step_num % self.restart == 0:
            self.prev_d = None
        if self.prev_d is None:
            d = g
        else:
            d = g + g.dot(g) / self.prev_g.dot(self.prev_g) * self.prev_d
        a = self.line_search(func, grad, x, d)
        self.prev_d = d
        self.prev_g = g
        self._step_num += 1
        return x + a * d

    def params(self):
        return {
            'line_search': self.line_search,
            'restart': self.restart
        }
