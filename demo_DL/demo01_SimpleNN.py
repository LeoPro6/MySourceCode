
import torch
import torch.nn as nn


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # 定义一个输入层到隐藏层的全连接层
        self.fc1 = nn.Linear(2, 2)
        # 定义一个隐藏层到输出层的全连接层
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        # 前向传播
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # 输出层
        return x


# 创建模型实例
model = SimpleNN()

# 打印模型
print(model)
