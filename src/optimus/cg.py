import numpy as np

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
            d = g + g.dot(g - self.prev_g) / self.prev_g.dot(self.prev_g) * self.prev_d
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


class BealePowellCG(Optimizer):

    def __init__(self, dim, line_search=ScalarLS(), c1=0.2, c2=0.8, c3=1.2):
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.dim = dim
        self.line_search = line_search

        self.k = 1
        self.t = 1
        self.dk1 = None
        self.gk1 = None
        self.dt = None
        self.gt = None
        self.gtp1 = None
        self.k = 0

    def params(self):
        return {
            'line_search': self.line_search,
            'c1': self.c1,
            'c2': self.c2,
            'c3': self.c3,
        }

    def step(self, x, func, grad):
        self.k += 1
        g = grad(x)
        if self.k == 2:
            self._reset(g)  # just to set up values properly
        if self.k == 1:
            d = - g
        else:
            if self.k - self.t >= self.dim or np.abs(self.gk1.dot(g)) > self.c1 * g.dot(g):
                self._reset(g)
                d = self._step(g)
            else:
                d = self._step(g)
                if self.k > self.t + 1 and not (
                        - self.c3 * g.dot(g) <= d.dot(g) <= - self.c2 * g.dot(g)
                ):
                    self._reset(g)
                    d = self._step(g)

        a = self.line_search(func, grad, x, d)
        self.dk1 = d
        self.gk1 = g
        return x + a * d

    def _step(self, gk):
        if self.k > self.t + 1:
            yk1 = gk - self.gk1
            yt = self.gtp1 - self.gt
            beta = gk.dot(yk1) / self.dk1.dot(yk1)
            gamma = gk.dot(yt) / self.dt.dot(yt)
            return - gk + beta * self.dk1 + gamma * self.dt
        elif self.k == self.t + 1:
            yk1 = gk - self.gk1
            beta = gk.dot(yk1) / self.dk1.dot(yk1)
            return - gk + beta * self.dk1
        raise Exception()

    def _reset(self, g):
        self.t = self.k - 1
        self.dt = self.dk1
        self.gt = self.gk1
        self.gtp1 = g
