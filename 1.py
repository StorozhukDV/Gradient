
import numpy as np
import matplotlib.pyplot as plt

class Classic():
    max_iter = 10000
    rate = 0.01
    epsilon = 10 ** -6
    def __init__(self):
        self.f = lambda x1, x2: 6 * x1 + x1 ** 2 - 4 * x2 + x2 ** 2
        self.funcgrad1 = lambda x1, x2: 6 + 2 * x1
        self.funcgrad2 = lambda x1, x2: -4 + 2 * x2
        self.GD_x1 = []
        self.GD_x2 = []

    def Spusk(self):
        x1 = 10
        x2 = 5
        f_change = self.f(x1, x2)
        iter = 0
        while f_change > self.epsilon and iter < self.max_iter:
            tmp_x1 = x1 + self.rate * self.f(x1, x2)
            tmp_x2 = x2 + self.rate * self.f(x1, x2)
            tmp_y = self.f(tmp_x1, tmp_x2)

            f_change = np.absolute(tmp_y - f_change)
            x1 = tmp_x1
            x2 = tmp_x2
            iter += 1
        print("Крайняя точка = ", (x1, x2))
        print("Экстремум = ", self.f(x1, x2))


Classic_gradient = Classic()
Classic_gradient.Spusk()
