"""
参考：https://www.zhihu.com/question/482747456
"""

import torch
from torch import nn


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.w1 = nn.Parameter(torch.tensor([2.], requires_grad=True))
        self.w2 = nn.Parameter(torch.tensor([3.], requires_grad=True))
        self.w3 = nn.Parameter(torch.tensor([4.], requires_grad=True))

    def forward(self, a):
        s = a * self.w1

        c = s * self.w2
        d = s * self.w3

        return c, d


x = torch.tensor([1.])
y1 = torch.tensor([10.])
y2 = torch.tensor([12.])

model = Model()
mse = torch.nn.MSELoss()
# optim = torch.optim.Adam(model.parameters(), lr=0.5)
optim = torch.optim.SGD(model.parameters(), lr=0.1)

num = 0
torch.autograd.set_detect_anomaly(True)  # For debugging.
while True:
    num += 1
    print(num)
    y1_pred, y2_pred = model(x)
    loss1 = mse(y1_pred, y1)
    loss2 = mse(y2_pred, y2)

    optim.zero_grad()
    loss1.backward(retain_graph=True)
    optim.step()

    optim.zero_grad()
    loss2.backward(retain_graph=True) # 报错行！while循环第二次运行到这里才会报错。为什么第一次不报错？
    optim.step()

"""
第二个backward()中，虽然w2没有更新，但是w2的version值还是增加了1。
这是因为w2在计算图中，系统认为它参与了更新，虽然值不变。
"""