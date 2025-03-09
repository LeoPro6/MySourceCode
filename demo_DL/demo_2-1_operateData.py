import torch

x = torch.arange(12)
print(x)
print(x.shape)
print(x.numel())

X = x.reshape(3, 4)
print(X)

print(torch.ones(2, 3, 4))

print(torch.randn(2, 3, 4))

xx = torch.tensor([[4, 1, 2, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(xx)

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print("({}\n {}\n {}\n {}\n {})\n".format(x + y, x - y, x * y, x / y, x ** y))  # **运算符是求幂运算

print(torch.exp(x))

# 线性代数
x_data = torch.arange(12, dtype=torch.float).reshape(3, 4)
y_data = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# 分别按行、按列连结两个矩阵
print("{}\n{}".format(torch.cat((x_data, y_data), dim=0), torch.cat((x_data, y_data), dim=1)))

print(x_data == y_data)
print(x_data.sum())

# 广播机制
a = torch.arange(3).reshape(3, 1)
b = torch.arange(2).reshape(1, 2)

print("{}\n{}".format(a, b))

print(a + b)

# 索引和切片
print("{}\n{}\n{}\n".format(x_data, x_data[1:3], x_data[1:2]))
x_data[2, 2] = 9
x_data[0:2, :] = 8
print(x_data)

# 节省内存
before = id(y_data)
y_data = y_data + x_data  # 浪费内存
print(id(y_data) == before)

z_data = torch.zeros_like(y_data)
print('id(z_data):', id(z_data))
z_data[:] = x_data + y_data  # 使用切片表示法将操作的结果分配给先前分配的数组
print('id(z_data):', id(z_data))

before = id(x_data)
x_data += y_data
print(id(x_data) == before)

# 转换为python其他对象
A = x_data.numpy()
B = torch.tensor(A, dtype=torch.float)
print("{}\n{}".format(type(A), type(B)))

a = torch.tensor([3.5])
print("{}\n{}\n{}\n{}".format(a, a.item(), float(a), int(a)))
