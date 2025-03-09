#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project: demo01_test 
@File   : demo_4-4-2.py
@Author : Leo Dalton
@Date   : 2025/3/8 上午9:24 
@Desc   :
"""

import numpy as np


def numerical_diff(func, x):
    h = 1e-4
    return (func(x + h) - func(x - h)) / (2 * h)


def func_x0(x0):
    return x0 ** 2 + 4.0 ** 2


res = numerical_diff(func_x0, 3.0)
print(res)
