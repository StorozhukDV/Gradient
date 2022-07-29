import math

import numpy as np
import sympy
import copy
import matplotlib.pyplot as plt


class Gradient:
    max_iter = 10000
    epsilon = 0.000001
    rate = 0.01

    def __init__(self):
        self.w = np.random.sample((2, 1))  # Инициализируем вектор искомых параметров случайными числами
        w1, w2 = sympy.symbols("w1,w2")

        exp =  w1 ** 2 +  w2 ** 2 + 6 * w1 - 4 * w2
        self.f = sympy.lambdify([w1, w2], exp, "numpy")
        self.dfw1 = sympy.lambdify([w1, w2], exp.diff(w1))
        self.dfw2 = sympy.lambdify([w1, w2], exp.diff(w2))
        self.grad_w = np.zeros(self.w.shape)  # Создаем массив для градиента
        self.counter = 0  # Создаем счетчик количества интераций

    def calculate(self):  # Метод для пересчета градиента
        self.grad_w[0] = self.dfw1(self.w[0], self.w[1])
        self.grad_w[1] = self.dfw2(self.w[0], self.w[1])

    def action(self):  # Тут описывается действия в конкретном алгоритме
        pass

    def is_continue(self, previous):
        if abs(self.f(self.w[0], self.w[1]) - previous) < Gradient.epsilon:
            return False

        elif self.counter >= GradClassic.max_iter:
            print(type(self))
            print("Решение не найдено. Превышено максимальное количество итераций")
            print(f"Ответ w1 = {self.w[0]:.3f}, w2 = {self.w[1]:.3f}")
            print(f"Значение функции f = {self.f(self.w[0], self.w[1]):.3f}")
            return False

        else:
            return True


class GradClassic(Gradient):

    def action(self):
        z0 = self.rate * self.grad_w[0]
        z1 = self.rate * self.grad_w[1]
        previous = self.f(z0,z1)
        self.calculate()

        while self.is_continue(previous):
            previous = copy.deepcopy(self.f(self.w[0],self.w[1]))
            self.w[0] -= self.rate * self.grad_w[0]
            self.w[1] -= self.rate * self.grad_w[1]
            self.calculate()
            self.counter += 1


class GradMomentum(Gradient):
    lamb = 0.5
    gamma = 1 - lamb
    eta = (1 - gamma) * Gradient.rate

    def __init__(self):
        super().__init__()
        self.v = np.zeros((2, 1))
        # self.v = np.random.sample((2, 1))  # Инициализируем промежуточный вектор случайными числами

    def action(self):
        z0 = self.rate * self.grad_w[0]
        z1 = self.rate * self.grad_w[1]
        previous = self.f(z0, z1)
        self.calculate()
        while self.is_continue(previous):
            previous = copy.deepcopy(self.f(self.w[0],self.w[1]))
            for i in range(len(self.v)):
                self.v[i] = self.v[i] * GradMomentum.gamma + GradMomentum.eta * self.grad_w[i]
                self.w[i] -= self.v[i]
            self.calculate()
            self.counter += 1


class GradNag(GradMomentum):

    def calculate(self):  # Метод для пересчета градиента
        self.grad_w[0] = self.dfw1(self.w[0] - GradMomentum.gamma * self.v[0],
                                   self.w[1] - GradMomentum.gamma * self.v[1])
        self.grad_w[1] = self.dfw2(self.w[0] - GradMomentum.gamma * self.v[0],
                                   self.w[1] - GradMomentum.gamma * self.v[1])


class GradRmsProp(GradMomentum):
    lamb = 0.02
    gamma = 1 - lamb

    def action(self):
        z0 = self.rate * self.grad_w[0]
        z1 = self.rate * self.grad_w[1]
        previous = self.f(z0, z1)
        self.calculate()
        while self.is_continue(previous):
            previous = copy.deepcopy(self.f(self.w[0],self.w[1]))
            for i in range(len(self.v)):
                self.v[i] = self.v[i] * GradRmsProp.gamma + (1 - GradRmsProp.gamma) * pow(self.grad_w[i], 2)
                self.w[i] -= 0.3 * self.grad_w[i] / math.sqrt(
                    self.v[i] + Gradient.epsilon)
            self.calculate()
            self.counter += 1


class GradAdaDelta(Gradient):
    alpha = 0.95

    def __init__(self):
        super().__init__()
        self.v = np.random.sample((2, 1))  # Инициализируем промежуточный вектор случайными числами
        self.delta1 = 0.03
        self.delta2 = 0.03

    def action(self):
        z0 = self.rate * self.grad_w[0]
        z1 = self.rate * self.grad_w[1]
        previous = self.f(z0, z1)
        self.calculate()
        while self.is_continue(previous):
            previous = copy.deepcopy(self.f(self.w[0],self.w[1]))
            for i in range(len(self.v)):
                self.v[i] = self.v[i] * GradAdaDelta.alpha + (1 - GradAdaDelta.alpha) * pow(self.grad_w[i], 2)
                self.delta1 = self.grad_w[i] * (math.sqrt(self.delta2) + Gradient.epsilon) / (
                        math.sqrt(self.v[i]) + Gradient.epsilon)
                self.delta2 = GradAdaDelta.alpha * self.delta2 + (1 - GradAdaDelta.alpha) * pow(self.delta1, 2)
                self.w[i] -= self.delta1
            self.calculate()
            self.counter += 1


class GradAdam(GradAdaDelta):
    lamb = 0.6
    gamma = 1 - lamb
    alpha = 0.999

    def __init__(self):
        super().__init__()
        self.g = np.random.sample((2, 1))

    def action(self):
        z0 = self.rate * self.grad_w[0]
        z1 = self.rate * self.grad_w[1]
        previous = self.f(z0, z1)
        self.calculate()
        self.counter = 1
        while self.is_continue(previous):
            previous = copy.deepcopy(self.f(self.w[0],self.w[1]))
            for i in range(len(self.v)):
                self.v[i] = self.v[i] * GradAdam.gamma + (1 - GradAdam.gamma) * self.grad_w[i]
                self.g[i] = GradAdam.alpha * self.g[i] + (1 - GradAdam.alpha) * pow(self.grad_w[i], 2)
                v = self.v[i] / (1 - pow(GradAdam.gamma, self.counter ))
                g = self.g[i] / (1 - pow(GradAdam.alpha, self.counter ))
                self.w[i] -= Gradient.rate * v / (math.sqrt(g) + Gradient.epsilon)
            self.calculate()
            self.counter += 1


data = list()
for i in range(0, 6):
    data.append(list())
for i in range(0, 100):
    test1 = GradClassic()
    test1.action()
    data[0].append(test1.counter)
    test2 = GradMomentum()
    test2.action()
    data[1].append(test2.counter)
    test3 = GradNag()
    test3.action()
    data[2].append(test3.counter)
    test4 = GradRmsProp()
    test4.action()
    data[3].append(test4.counter)
    test5 = GradAdaDelta()
    test5.action()
    data[4].append(test5.counter)
    test6 = GradAdam()
    test6.action()
    data[5].append(test6.counter)

for i in range(len(data)):
    print(f"Среднее значение: {sum(data[i]) // len(data[i])}")

name_methods = ["Classic", "Momentum", "NAG", "RMSProp", "AdaDelta", "Adam"]
fig, ax = plt.subplots()
ax.boxplot(data)
ax.grid()
ax.set(
    axisbelow=True,

    xlabel='Метод',
    ylabel='Кол-во итераций', )
ax.set_xticklabels(np.repeat(name_methods, 1),
                   rotation=45, fontsize=8)
plt.show()

