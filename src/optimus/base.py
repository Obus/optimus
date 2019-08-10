import abc


class Parametrizable(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def params(self):
        raise NotImplementedError()

    def __repr__(self):
        return '{class_}({params})'.format(
            class_=self.__class__.__name__,
            params=', '.join(f'{k}={v}' for k, v in self.params().items())
        )


class Optimizer(Parametrizable, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def step(self, func, grad, x):
        raise NotImplementedError()
