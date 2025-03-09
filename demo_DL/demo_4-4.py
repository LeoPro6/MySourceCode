#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project: demo01_test 
@File   : demo_4-4.py
@Author : Leo Dalton
@Date   : 2025/3/7 下午9:52 
@Desc   :
"""

import numpy as np
import matplotlib.pyplot as plt


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp = x[idx]
        # f(x+h)的计算
        x[idx] = tmp + h
        fxh1 = f(x)

        # f(x-h)的计算
        x[idx] = tmp - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp

        it.iternext()

    return grad


def function_2(x):
    return x[0] ** 2 + x[1] ** 2


def gradient_decent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x = x - lr * grad

    return x


def softmax(a):
    c = np.max(a)
    return np.exp(a - c) / np.sum(np.exp(a - c))


def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


# print(numerical_gradient(function_2, np.array([3.0, 4.0])))
# print(numerical_gradient(function_2, np.array([0.0, 2.0])))
# print(numerical_gradient(function_2, np.array([3.0, 0.0])))
# print(gradient_decent(function_2, np.array([-3.0, 4.0]), 0.1, 100))

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


net = simpleNet()
print(net.W)

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p, np.argmax(p))

t = np.array([0, 0, 1])
print(net.loss(x, t))

f = lambda w: net.loss(x, t)

dW = numerical_gradient(f, net.W)
print(dW)
