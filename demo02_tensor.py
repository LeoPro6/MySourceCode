import torch
import numpy as np

data = [[1, 2], [3, 4]]

x_data = torch.tensor(data)
print(f"x_data: \n{x_data}\n")

np_array = np.array(data)
print(f"np_array: \n{np_array}\n")

x_np = torch.tensor(np_array)
print(f"x_np: \n{x_np}\n")

x_ones = torch.ones_like(x_data)
print(f"x_ones: \n{x_ones}\n")

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"x_rand: \n{x_rand}\n")

shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tesnor = torch.ones(shape)
zeros_tesnor = torch.zeros(shape)

print(f"rand_tensor: \n{rand_tensor}\n")
print(f"ones_tesnor: \n{ones_tesnor}\n")
print(f"zeros_tesnor: \n{zeros_tesnor}\n")

tensor = torch.rand(3, 4)
print(f"Shape of tensor: \n{tensor.shape}\n")
print(f"Datatype of tensor: \n{tensor.dtype}\n")
print(f"Device tensor is stored on: \n{tensor.device}\n")

if torch.cuda.is_available():
    tensor = tensor.to('cuda')
    print(f"Device tensor is stored on: \n{tensor.device}\n")

tensor = torch.ones(4, 4)
tensor[:, 1] = 0
print(tensor)

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(f"t1: \n{t1}")

x = torch.arange(12)
print(f"x: \n{x}\n")

x2 = x.reshape(3, 4)
print(f"x2: \n{x2}\n")