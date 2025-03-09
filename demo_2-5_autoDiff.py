import torch

x = torch.arange(4.0)
x.requires_grad_(True)

y = 2 * torch.dot(x, x)
print("{}\n{}\n{}\n{}".format(y, y.backward(), x.grad, x.grad == 4 * x))

x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)

# 分离计算
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x
z.sum().backward()
print(x.grad == u)


def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c


a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
print(d / a, a.grad == d / a)
