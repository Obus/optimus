from .base import Optimizer


class GD(Optimizer):

    def __init__(self, step_rate):
        self.step_rate = step_rate

    def step(self, x, func, grad):
        g = grad(x)
        return x - self.step_rate * g

    def params(self):
        return {
            'step_rate': self.step_rate,
        }
