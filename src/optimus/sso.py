import numpy as np
import scipy
import scipy.optimize
import scipy.stats
import scipy.sparse


from .base import Optimizer
from .line_search import ScalarLS


l2 = np.linalg.norm


def normalize(x):
    return x / np.sqrt(x.dot(x))


def orthogonalize(x, p):
    if p.dot(p) == 0:
        return x
    return x - p.dot(x) * p / p.dot(p)


def update_subspace1(D, g, subspace_dim=None):
    g = g.reshape((g.shape[0], 1))
    g /= np.sqrt(g.T.dot(g))
    if D.shape[1] == 0:
        return g

    if subspace_dim and D.shape[1] >= subspace_dim:
        D = D[:, -subspace_dim:]
    return np.hstack((D, g))


def update_subspace2(D, g, subspace_dim=None):
    for i in range(D.shape[1]):
        g = orthogonalize(g, D[:, i])
    g = normalize(g)
    if subspace_dim and D.shape[1] >= subspace_dim:
        D = D[:, -subspace_dim:]
    return np.hstack((D, g.reshape((g.shape[0], 1))))


def update_subspace3(D, g, subspace_dim=None):
    g = normalize(g)
    for i in range(D.shape[1]):
        D[:, i] = normalize(orthogonalize(D[:, i], g))

    if subspace_dim and D.shape[1] >= subspace_dim:
        D = D[:, -subspace_dim:]
    return np.hstack((D, g.reshape((g.shape[0], 1))))


def update_P_cg(P, g1, g0):
    #     b_k = g1.dot(g1) / g0.dot(g0)
    b_k = g1.dot(g1) / (g0).dot(g0)
    p_k = - g1 + P[-1, :] * b_k
    return np.vstack((P[1:, :], normalize(p_k)))


g0 = None
p0 = None


def update_subspace4(D, g, subspace_dim=None):
    global g0
    global p0
    if p0 is None:
        g0 = g
        p0 = -g
        return normalize(g).reshape((g.shape[0], 1))

    g1 = g
    b1 = g1.dot(g1) / (g0).dot(g0)
    p1 = - g1 + p0 * b1

    g0 = g1
    p0 = p1

    for i in range(D.shape[1]):
        D[:, i] = normalize(orthogonalize(D[:, i], p1))

    if subspace_dim and D.shape[1] >= subspace_dim:
        D = D[:, -subspace_dim:]
    return np.hstack((D, normalize(p1).reshape((g.shape[0], 1))))


update_subspace = update_subspace2


class SSO_ideal(Optimizer):

    def __init__(self, x_optim, dim, subspace_dim, subspace_update=update_subspace2):
        self.x_optim = x_optim
        self.dim = dim
        self.subspace_dim = subspace_dim
        self.subspace_update = subspace_update
        self.D = np.zeros((dim, 0))

    def step(self, x, func, grad):
        g = grad(x)
        self.D = self.subspace_update(self.D, g, subspace_dim=self.subspace_dim)
        D = self.D
        return self.x_optim + (np.eye(self.dim) - D.dot(D.T)).dot(x - self.x_optim)

    def params(self):
        return {
            'subspace_dim': self.subspace_dim,
            'subspace_update': self.subspace_update,
        }


class SSO_ideal_linea_search(Optimizer):

    def __init__(
            self,
            A,
            dim,
            subspace_dim,
            subspace_update=update_subspace2,
            line_search=ScalarLS(),
            add_grad_proj=False
    ):
        self.A = A
        self.dim = dim
        self.subspace_dim = subspace_dim
        self.subspace_update = subspace_update
        self.add_grad_proj = add_grad_proj
        self.line_search = line_search
        self.D = np.zeros((dim, 0))

    def step(self, x, func, grad):
        g = grad(x)
        self.D = self.subspace_update(self.D, g, subspace_dim=self.subspace_dim)
        D = self.D
        try:
            d = D @ np.linalg.inv(D.T @ self.A @ D) @ D.T @ g
            if self.add_grad_proj:
                d = d + g - D @ D.T @ g
        except Exception as e:
            #             print(D)
            raise
        if any(np.isnan(d)):
            d[np.isnan(d)] = 0
        #             raise Exception(f'NaN occured in d: {d}')
        a = self.line_search(func, grad, x, d)
        x = x + a * d
        return x

    def params(self):
        return {
            'subspace_dim': self.subspace_dim,
            'subspace_update': self.subspace_update,
            'line_search': self.line_search,
            'add_grad_proj': self.add_grad_proj,
        }


class SecantSystemSolver(Optimizer):

    def __init__(
            self,
            dim,
            line_search=ScalarLS(),
            subspace_dim=None,
    ):
        self.dim = dim
        self.subspace_dim = subspace_dim
        self.line_search = line_search

        self.g_trace = []
        self.x_trace = []

    def step(self, x, func, grad):
        g = grad(x)
        self.x_trace.append(x)
        self.g_trace.append(g)
        if len(self.g_trace) > 1:
            if self.subspace_dim:
                g_trace = self.g_trace[-self.subspace_dim:]
                x_trace = self.x_trace[-self.subspace_dim:]
            else:
                g_trace = self.g_trace
                x_trace = self.x_trace
            G = np.vstack(
                [(g - g_) * l2(g) ** 2 / l2(g_) ** 2 for i, g_ in enumerate(g_trace[:-1])])
            y = np.array([g.dot(x - x_) for x_ in x_trace[:-1]])
            d = scipy.sparse.linalg.lsqr(G, y, atol=1e-30, btol=1e-30)[0]
        else:
            d = g
        a = self.line_search(func, grad, x, d)
        x = x + a * d
        return x

    def params(self):
        return {
            'line_search': self.line_search,
            'subspace_dim': self.subspace_dim,
        }


class SecantSystemSolver2(Optimizer):

    def __init__(
            self,
            dim,
            line_search=ScalarLS(),
            subspace_dim=None,
            step_rate=1e-4,
    ):
        self.dim = dim
        self.subspace_dim = subspace_dim
        self.line_search = line_search
        self.step_rate = step_rate

        self.g_trace = []
        self.x_trace = []
        self.pred_d = None
        self.a_trace = []
        self.d_newton_trace = []
        self.YYdcg_trace = []

    def step(self, x, func, grad):
        g = grad(x)
        self.x_trace.append(x)
        self.g_trace.append(g)
        if len(self.g_trace) > 1:
            if self.subspace_dim:
                g_trace = self.g_trace[-self.subspace_dim:]
                x_trace = self.x_trace[-self.subspace_dim:]
            else:
                g_trace = self.g_trace
                x_trace = self.x_trace
            #             G = np.vstack([(g - g_) * l2(g) ** 2 / l2(g_) ** 2 for i, g_ in enumerate(g_trace[:-1])])
            G = np.vstack([(g - g_) for i, g_ in enumerate(g_trace[:-1])])
            Z = np.vstack([(x - x_) for x_ in x_trace[:-1]])
            GG = G.T @ np.linalg.pinv(G @ G.T) @ G
            #             y = np.array([g.dot(x - x_)  * l2(g) ** 2 / l2(g_) ** 2  for x_, g_ in zip(x_trace[:-1], g_trace[:-1])])
            y = np.array([g.dot(x - x_) for x_, g_ in zip(x_trace[:-1], g_trace[:-1])])
            d_newton = scipy.sparse.linalg.lsqr(G, y, atol=1e-30, btol=1e-30)[0]
            self.d_newton_trace.append(d_newton)
            #             z_grad = g * 0.5 + self.pred_d * 0.5
            z_grad = g + g_trace[-1].dot(g_trace[-1]) / g_trace[-2].dot(g_trace[-2]) * self.pred_d
            self.YYdcg_trace.append(l2(GG @ z_grad) / l2(z_grad))
            d_grad = (np.eye(self.dim) - GG) @ z_grad
            #             d_grad = z_grad
            #             d_grad =  (np.eye(self.dim) - G.T @ np.linalg.pinv(G @ G.T) @ G) @ z_grad
            d = d_grad + d_newton

        #             if l2(d_newton) > 0:
        #                 a_newton = self.line_search(func, grad, x, d_newton)
        #                 x = x + a_newton * d_newton
        #                 d = (np.eye(self.dim) - ZZ) @ z_grad
        #             else:
        #                 d = z_grad

        else:
            d = g
        a = self.line_search(func, grad, x, d)
        self.a_trace.append(a)
        x = x + a * d
        self.pred_d = d
        return x

    def params(self):
        return {
            'line_search': self.line_search,
            'subspace_dim': self.subspace_dim,
            'step_rate': self.step_rate,
        }


class MixedSecantSystemSolver(Optimizer):

    def __init__(
            self,
            dim,
            line_search=ScalarLS(),
            subspace_dim=None,
            w=0.99999999
    ):
        self.dim = dim
        self.subspace_dim = subspace_dim
        self.line_search = line_search
        self.w = w

        self.g_trace = []
        self.x_trace = []
        self.prev_d = None
        #         self.step = 0
        self.a_trace = []

    def step(self, x, func, grad):
        #         self.step += 1
        g = grad(x)
        self.x_trace.append(x)
        self.g_trace.append(g)
        if len(self.g_trace) > 1:
            if self.subspace_dim:
                g_trace = self.g_trace[-self.subspace_dim:]
                x_trace = self.x_trace[-self.subspace_dim:]
            else:
                g_trace = self.g_trace
                x_trace = self.x_trace
            G = np.vstack([(g - g_) * l2(g) ** 2 / l2(g_) ** 2 for g_ in g_trace[:-1]])
            y = np.array([g.dot(x - x_) for x_ in x_trace[:-1]])
            w = self.w
            d = np.linalg.pinv(
                w * G.T @ G + (1 - w) * np.eye(G.shape[1])
            ).dot(
                w * G.T.dot(y)
                + (1 - w) * (self.prev_d + g) / 2
            )
        else:
            d = g
        a = self.line_search(func, grad, x, d)
        self.a_trace.append(a)
        x = x + a * d
        self.prev_d = d
        return x

    def params(self):
        return {
            'line_search': self.line_search,
            'subspace_dim': self.subspace_dim,
            'w': self.w
        }

    
def update_subspace2(D, g, subspace_dim=None):
    for i in range(D.shape[1]):
        g = orthogonalize(g, D[:, i])
    g = normalize(g)
    if subspace_dim and D.shape[1] >= subspace_dim - 1:
        D = D[:, -(subspace_dim - 1):]
    return np.hstack((D, g.reshape((g.shape[0], 1))))
    
    
class LQNSSO_CG(Optimizer):

    def __init__(
            self,
            dim,
            line_search=ScalarLS(),
            subspace_update=update_subspace2,
            subspace_dim=None,
            step_rate=1e-4,
    ):
        self.dim = dim
        self.subspace_dim = subspace_dim
        self.subspace_update = subspace_update
        self.line_search = line_search
        self.step_rate = step_rate

        self.D = np.zeros((dim, 0))
        self.g_trace = []
        self.x_trace = []
        self.pred_d = None

        self._debug = {
            'a_trace': [],
        }

    def step(self, x, func, grad):
        g = grad(x)
        # self.D = self.subspace_update(self.D, g, subspace_dim=self.subspace_dim)
        # D = self.D
        self.x_trace.append(x)
        self.g_trace.append(g)
        if len(self.g_trace) > 1:
            if self.subspace_dim:
                g_trace = self.g_trace[-self.subspace_dim:]
                x_trace = self.x_trace[-self.subspace_dim:]
            else:
                g_trace = self.g_trace
                x_trace = self.x_trace
            G = np.vstack([(g - g_) for g_ in g_trace[:-1]])
#             GG = G.T @ np.linalg.pinv(G @ G.T) @ G
            y = np.array([g.dot(x - x_) for x_ in x_trace[:-1]])
            if l2(y) == 0:
                d_newton = np.zeros(self.dim)
            else:
                d_newton = scipy.sparse.linalg.lsqr(G, y, atol=1e-30, btol=1e-30)[0]
            z_grad = g + g.dot(g) / g_trace[-2].dot(g_trace[-2]) * self.pred_d
            if l2(y) == 0:
                d_grad = z_grad
            else:
                d_grad = z_grad - G.T @ (np.linalg.pinv(G @ G.T) @ (G @ z_grad))
            d = d_grad + d_newton
        else:
            d = g
        a = self.line_search(func, grad, x, d)
        self._debug['a_trace'].append(a)
        x = x + a * d
        self.pred_d = d
        return x

    def params(self):
        return {
            'line_search': self.line_search,
            'subspace_dim': self.subspace_dim,
#             'subspace_update': self.subspace_update,
            'step_rate': self.step_rate,
        }
