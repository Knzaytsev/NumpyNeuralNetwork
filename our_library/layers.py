import numpy as np


class Sigmoid:
    def __init__(self):

        self._sigmoid_result = None

    def forward(self, x):

        self._sigmoid_result = 1 / (1 + np.exp(-x))

        return self._sigmoid_result

    def backward(self, grad):

        new_grad = self._sigmoid_result * (1 - self._sigmoid_result) * grad

        return new_grad

    def step(self, learning_step):

        pass


class NLLLoss:
    def __init__(self):

        self._softmax_result = None
        self._y = None

    @staticmethod
    def softmax(x, axis=1):

        exp_scores = np.exp(x)

        return exp_scores / exp_scores.sum(axis, keepdims=True)

    def forward(self, x, y):

        self._softmax_result = self.softmax(x)

        self._y = np.zeros_like(x)
        self._y[np.arange(x.shape[0]), y] = 1

        loss = -(np.log(self._softmax_result) * self._y).sum(1).mean()

        return loss

    def backward(self):

        return (self._softmax_result - self._y) / self._y.shape[0]

    def step(self, learning_rate):

        pass

class Linear:

    def __init__(self, in_features, out_features):
        
        # инициализируем веса
        self.weight = np.random.rand(in_features, out_features) * 0.01
        self.bias = np.zeros(out_features)
        
        # сохраняем градиент для весов
        self._grad_weight = None
        self._grad_bias = None
        
        self._x = None
        self._linear_result = None

    def forward(self, x):
        self._x = np.copy(x)
        self._linear_result = np.dot(self._x, self.weight) + self.bias
        return self._linear_result

    def backward(self, grad):
        self._grad_weight = np.dot(self._x.T, grad)
        self._grad_bias = np.dot(np.ones((self._x.shape[0])), grad)
        return np.dot(grad, self.weight.T)

    def step(self, learning_rate):
        self.weight -= learning_rate * self._grad_weight
        self.bias -= learning_rate * self._grad_bias
        
class ReLU:

    def __init__(self):
        self._relu_result = None

    def forward(self, x):
        self._relu_result = np.maximum(x, 0)
        return self._relu_result

    def backward(self, prev_grad):
        grad = np.copy(prev_grad)
        grad[self._relu_result == 0] = 0
        return grad

    def step(self, learning_rate):
        pass
    
class BCELoss:
    def __init__(self):
        self._y_hat = None
        self._y = None
        self._bce_result = None
    
    def forward(self, y_hat, y):
        self._y_hat = np.copy(y_hat)
        self._y = np.expand_dims(y, 1)
        self._bce_result = -1 / self._y_hat.shape[0] * np.sum(self._y.T.dot(np.log(self._y_hat)) + (1 - self._y).T.dot(np.log(1 - self._y_hat)))
        return self._bce_result

    def backward(self):
        return 1 / self._y_hat.shape[0] * (-(self._y / self._y_hat) + (1 - self._y) / (1 - self._y_hat))

    def step(self, learning_rate):
        pass
    
class NeuralNetwork:

    def __init__(self, modules):

        # список слоев
        self.modules = modules

    def forward(self, x):
        forward = np.copy(x)
        for module in self.modules:
            forward = module.forward(forward)
        return forward

    def backward(self, grad):
        """
        :grad: градиент от функции потерь
        :return: возвращать ничего не потребуется
        """
        for module in self.modules[::-1]:
            grad = module.backward(grad)

    def step(self, learning_rate):
        for module in self.modules:
            module.step(learning_rate)