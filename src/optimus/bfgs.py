import numpy as np

from .base import Optimizer
from .line_search import ScalarLS


class BFGS(Optimizer):

    def __init__(self, dim, line_search=ScalarLS()):
        self.dim = dim
        self.line_search = line_search
        self.H = np.eye(dim)

    def step(self, x, func, grad):
        g = grad(x)
        d = - np.dot(self.H, g)

        a = self.line_search(func, grad, x, d)

        x1 = x + a * d
        s = x1 - x
        g1 = grad(x1)
        y = g1 - g
        ro = 1.0 / (np.dot(y, s))
        if ro < 0:
            return x + self.line_search(func, grad, x, g) * g
        I = np.eye(self.dim)
        A1 = I - ro * s[:, np.newaxis] * y[np.newaxis, :]
        A2 = I - ro * y[:, np.newaxis] * s[np.newaxis, :]
        self.H = np.dot(A1, np.dot(self.H, A2)) + (ro * s[:, np.newaxis] * s[np.newaxis, :])

        return x1

    def params(self):
        return {
            'line_search': self.line_search,
        }


class LBFGS(Optimizer):

    def __init__(self, dim, subspace_dim, line_search=ScalarLS()):
        self.dim = dim
        self.subspace_dim = subspace_dim
        self.line_search = line_search
        self.x_trace = []
        self.g_trace = []

    def step(self, x, func, grad):
        g = grad(x)
        self.x_trace.append(x)
        self.g_trace.append(g)
        if len(self.x_trace) > self.subspace_dim:
            self.x_trace = self.x_trace[-self.subspace_dim:]
            self.g_trace = self.g_trace[-self.subspace_dim:]
        if len(self.x_trace) < 2:
            d = -g
        else:
            y_trace = [g1 - g0 for g1, g0 in zip(self.g_trace[1:], self.g_trace[:-1])]
            s_trace = [x1 - x0 for x1, x0 in zip(self.x_trace[1:], self.x_trace[:-1])]
            q = g
            alphas = []
            for y, s in zip(reversed(y_trace), reversed(s_trace)):
                rho = 1 / y.dot(s)
                alpha = rho * s.dot(q)
                q = q - alpha * y
                alphas.append(alpha)
            y = y_trace[-1]
            s = s_trace[-1]
            gamma = s.dot(y) / y.dot(y)
            z = gamma * q

            for y, s, alpha in zip(y_trace, s_trace, reversed(alphas)):
                rho = 1 / y.dot(s)
                beta = rho * y.dot(z)
                z = z + s * (alpha - beta)
            d = z
        a = self.line_search(func, grad, x, d)
        return x + a * d

    def params(self):
        return {
            'line_search': self.line_search,
            'subspace_dim': self.subspace_dim,
        }