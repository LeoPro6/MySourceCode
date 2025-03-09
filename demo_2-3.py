import torch

# 标量
x = torch.tensor(3.0)
y = torch.tensor(2.0)
print("{}\n{}\n{}\n{}\n".format(x + y, x * y, x / y, x ** y))

# 向量
x = torch.arange(4, dtype=torch.float32)
print("{}\n{}\n{}\n{}".format(x, x[2], len(x), x.shape))

# 矩阵
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
print("{}\n{}\n".format(A, A.T))
# 对称矩阵
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print("{}\n{}\n".format(B, B == B.T))

# 降维
A_sum_axis0 = A.sum(axis=0)
A_sum_axis1 = A.sum(axis=1)
print("axis0: {}\n{}\naxis1: {}\n{}\n".format(A_sum_axis0, A_sum_axis0.shape,
                                              A_sum_axis1, A_sum_axis1.shape))
print("axis0-1: {}".format(A.sum(axis=[0, 1])))

print("mean:{}\nsum/numel:{}\n".format(A.mean(), A.sum() / A.numel()))

# 非降维求和
sum_A = A.sum(axis=1, keepdims=True)
print(sum_A, A / sum_A)

# 点积：相同位置的元素乘积的和
y = torch.ones(4, dtype=torch.float32)
print("{}\n{}\n{}\n".format(x, y, torch.dot(x, y)))

# 向量积
print("{}\nZ{}\n{}".format(A.shape, x.shape, torch.mv(A, x)))

# 矩阵乘法
B = torch.ones(4, 3)
print("{}\n".format(torch.mm(A, B)))
