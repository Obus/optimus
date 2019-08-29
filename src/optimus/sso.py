import numpy as np
import scipy
import scipy.optimize
import scipy.stats
import scipy.sparse


from .base import Optimizer
from .line_search import ScalarLS


l2 = np.linalg.norm


def normalize(x):
    l2x = l2(x)
    if l2x > 0:
        return x / l2x
    return x


def orthogonalize(x, p):
    if p.dot(p) == 0:
        return x
    return x - p.dot(x) * p / p.dot(p)


def update_subspace1(D, g, subspace_dim=None):
    """columns of D - normalized gradients (not orthogonalized)"""
    g = normalize(g).reshape((g.shape[0], 1))
    if D.shape[1] == 0:
        return g

    if subspace_dim and D.shape[1] >= subspace_dim:
        D = D[:, -subspace_dim:]
    return np.hstack((D, g))


def update_subspace2(D, g, subspace_dim=None):
    """columns of D - orthonormalized gradients,
    g is orthogonalized with respect to existing once"""
    for i in range(D.shape[1]):
        g = orthogonalize(g, D[:, i])
    g = normalize(g)
    if subspace_dim and D.shape[1] >= subspace_dim - 1:
        D = D[:, -(subspace_dim - 1):]
    return np.hstack((D, g.reshape((g.shape[0], 1))))


def update_subspace3(D, g, subspace_dim=None):
    """columns of D - orthonormalized gradients,
    existing columns are orthogonalized with respect to g"""
    g = normalize(g)
    for i in range(D.shape[1]):
        D[:, i] = normalize(orthogonalize(D[:, i], g))

    if subspace_dim and D.shape[1] >= subspace_dim - 1:
        D = D[:, - (subspace_dim - 1):]
    return np.hstack((D, g.reshape((g.shape[0], 1))))
    
    
class LQNSSO_CG(Optimizer):

    def __init__(
            self,
            dim,
            line_search=ScalarLS(),
            subspace_update=update_subspace2,
            subspace_dim=None,
    ):
        self.dim = dim
        self.subspace_dim = subspace_dim
        self.subspace_update = subspace_update
        self.line_search = line_search

        self.D = np.zeros((dim, 0))
        self.g_trace = []
        self.x_trace = []
        self.pred_d = None

    def step(self, x, func, grad):
        g = grad(x)
        self.D = self.subspace_update(self.D, g, subspace_dim=self.subspace_dim)
        D = self.D
        self.x_trace.append(x)
        self.g_trace.append(g)
        if self.subspace_dim:
            self.g_trace = self.g_trace[-self.subspace_dim:]
            self.x_trace = self.x_trace[-self.subspace_dim:]
        if len(self.g_trace) > 1:
            g_trace = self.g_trace
            x_trace = self.x_trace
            G = np.vstack([D.T.dot(g - g_) for g_ in g_trace[:-1]])
            y = np.array([g.dot(x - x_) for x_ in x_trace[:-1]])
            if l2(y) == 0:
                d_newton = np.zeros(self.subspace_dim)
            else:
                d_newton = scipy.sparse.linalg.lsqr(G, y, atol=1e-30, btol=1e-30)[0]
            d_grad = g + g.dot(g) / g_trace[-2].dot(g_trace[-2]) * self.pred_d
            if l2(y) > 0:
                d_grad = d_grad - D @ G.T @ (np.linalg.pinv(G @ G.T) @ (G @ D.T @ d_grad))
            d = d_grad + D @ d_newton
        else:
            d = g
        a = self.line_search(func, grad, x, d)
        x = x + a * d
        self.pred_d = d
        return x

    def params(self):
        return {
            'line_search': self.line_search,
            'sd': self.subspace_dim,
        }


class CustomizableLQNSSO_CG(Optimizer):

    def __init__(
            self,
            dim,
            line_search=ScalarLS(),
            subspace_update=update_subspace2,
            subspace_dim=None,
            use_D=True,
            remove_G=True,
            min_y=0,
            restart=None,
    ):
        self.dim = dim
        self.subspace_dim = subspace_dim
        self.subspace_update = subspace_update
        self.line_search = line_search
        self.use_D = use_D
        self.remove_G = remove_G
        self.min_y = min_y
        self.restart = restart

        self.D = np.zeros((dim, 0))
        self.g_trace = []
        self.x_trace = []
        self.pred_d = None
        self._step = 0

    def step(self, x, func, grad):
        self._step += 1

        if self.restart and self._step % self.restart == 0:
            self.D = np.zeros((self.dim, 0))
            self.g_trace = []
            self.x_trace = []
            self.pred_d = None

        g = grad(x)
        self.D = self.subspace_update(self.D, g, subspace_dim=self.subspace_dim)
        D = self.D
        self.x_trace.append(x)
        self.g_trace.append(g)
        if self.subspace_dim:
            self.g_trace = self.g_trace[-self.subspace_dim:]
            self.x_trace = self.x_trace[-self.subspace_dim:]
        if len(self.g_trace) > 1:
            g_trace = self.g_trace
            x_trace = self.x_trace

            if self.use_D:
                G = np.vstack([D.T.dot(g - g_) for g_ in g_trace[:-1]])
            else:
                G = np.vstack([(g - g_) for g_ in g_trace[:-1]])
            y = np.array([g.dot(x - x_) for x_ in x_trace[:-1]])
            if l2(y) / l2(g) <= self.min_y:
                if self.use_D:
                    d_newton = D @ np.zeros(self.subspace_dim)
                else:
                    d_newton = np.zeros(self.dim)
            else:
                if self.use_D:
                    d_newton = D @ scipy.sparse.linalg.lsqr(G, y, atol=1e-30, btol=1e-30)[0]
                else:
                    d_newton = scipy.sparse.linalg.lsqr(G, y, atol=1e-30, btol=1e-30)[0]
            z_grad = g + g.dot(g) / g_trace[-2].dot(g_trace[-2]) * self.pred_d
            if l2(y) / l2(g) <= self.min_y:
                d_grad = z_grad
            else:
                if self.use_D:
                    z_grad_G = D @ G.T @ (np.linalg.pinv(G @ G.T) @ (G @ D.T @ z_grad))
                else:
                    z_grad_G = G.T @ (np.linalg.pinv(G @ G.T) @ (G @ z_grad))
                if self.remove_G:
                    d_grad = z_grad - z_grad_G
                else:
                    d_grad = z_grad

            d = d_grad + d_newton
        else:
            d = g
        a = self.line_search(func, grad, x, d)
        x = x + a * d
        self.pred_d = d
        return x

    def params(self):
        return {
            'sd': self.subspace_dim,
            'rm_G': self.remove_G,
            'use_D': self.use_D,
            'restart': self.restart,
            'min_y': self.min_y,
        }
