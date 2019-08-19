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


class BealePowellCG(Optimizer):

    def __init__(self, dim, line_search=ScalarLS(), c1=0.2, c2=0.9, c3=1.2):
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.dim = dim
        self.line_search = line_search

        self.k = 1
        self.t = 1
        self.d1 = None
        self.d2 = None
        self.g1 = None
        self.g2 = None
        self.dt = None
        self.dt1 = None
        self.gt = None
        self.gt1 = None

    def params(self):
        return {
            'line_search': self.line_search,
            'dim': self.dim,
            #             'c1': self.c1,
            #             'c2': self.c2,
            #             'c3': self.c3,
            #             'subspace_update': self.subspace_update,
            #             'step_rate': self.step_rate,
        }

    def step(self, x, func, grad):
        g = grad(x)
        if self.k == 1:
            d = - g
        else:
            # pass
            if self.k - self.t >= self.dim or np.abs(self.d1.dot(g)) > self.c1 * g.dot(g):
                self._reset()
                d = self._step(g)
            else:
                d = self._step(g)
                if not (- self.c3 * g.dot(g) <= d.dot(g) <= - self.c2 * g.dot(g)):
                    self._reset()
                    d = self._step(g)

        a = self.line_search(func, grad, x, d)
        self.d1 = d
        self.d2 = self.d1
        self.g1 = g
        self.g2 = self.g1
        return x + a * d

    def _step(self, g):
        if self.k > self.t + 1:
            yk = g - self.g1
            yt = self.gt - self.gt1
            beta = g.dot(yk) / self.d1.dot(yk)
            gamma = g.dot(yt) / self.dt1.dot(yt)
            return - g + beta * self.d1 + gamma * self.dt
        elif self.k == t + 1:
            yk = g - self.g1
            beta = g.dot(yk) / self.d1.dot(yk)
            return - g + beta * self.d1
        else:
            raise Exception()

    def _reset(self):
        self.t = self.k - 1
        self.dt = self.d1
        self.dt1 = self.d2
        self.gt = self.g1
        self.gt1 = self.g2
