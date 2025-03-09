#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project: demo01_test 
@File   : demo_5-4.py
@Author : Leo Dalton
@Date   : 2025/3/8 下午1:07 
@Desc   :
"""

import numpy as np


class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


apple_price = 100
apple_num = 2
apple_tax = 1.1

mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# forward
apple_total_price = mul_apple_layer.forward(apple_price, apple_num)
pay_price = mul_tax_layer.forward(apple_total_price, apple_tax)

print(pay_price)

dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(dapple, dapple_num, dtax)