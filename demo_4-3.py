#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project: demo01_test
@File   : demo_4-3.py
@Author : Leo Dalton
@Date   : 2025/3/7 下午9:13
@Desc   :
"""

import numpy as np
import matplotlib.pyplot as plt


def numerical_diff(f, x):
    h = 1e-4  # 0.0001
    return (f(x + h) - f(x - h)) / (2 * h)


def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x


x = np.arange(0.0, 20.0, 0.1)  # 以0.1为步进，从0到20的数组x
y = function_1(x)

print(numerical_diff(function_1, 5))
print(numerical_diff(function_1, 10))

plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show()
