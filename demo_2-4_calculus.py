import torch
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return 3 * x ** 2 - 4 * x


def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h


h = 0.1
for i in range(10):
    print(f'h={h:.8f}, numbreical limit={numerical_lim(f, 1, h):.8f}')
    h *= 0.1
