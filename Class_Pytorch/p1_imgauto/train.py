import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import Class_OS.o1_获得当前工作目录

# 参数列表
EPOCH = 30
BATCH_SIZE = 512
LR = 0.01
IMG_H = 30
IMG_W = 15
path = Class_OS.o1_获得当前工作目录.main()


# 加载数据
class ReaderData(Dataset):
    def __init__(self, path):
        with open(path + "data/ocrData.txt", 'rt') as f:
            self.a = f.read()

    def __getitem__(self, index):
        im = Image.open(path + "data/" + str(index) + ".jpg").convert('L')
        im = np.array(im).reshape(IMG_H, IMG_W).astype(np.float32)
        im = im / 255.0 * 2.0 - 1.0
        # im = torch.from_numpy(im)
        return im

    def __len__(self):
        return len(self.a)


# 数据读取
train_data = ReaderData(path=path)
data_loader = DataLoader(train_data, batch_size=512)

# 定义网络
"""
抄莫烦的自解码网络
"""


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(IMG_H * IMG_W, 256),
        )
        # 解压
        self.decoder = nn.Sequential(
            nn.Linear(256, IMG_H * IMG_W),
            nn.Sigmoid(),  # 激励函数让输出值在 (0, 1)
        )
        self.enZ = nn.Sequential(
            nn.Linear(256, 3),  # 压缩成3个特征值方便画图
        )

    def forward(self, x):
        encoded = self.encoder(x)
        enZ = self.enZ(encoded)
        decoded = self.decoder(encoded)

        return enZ, decoded


# 定义训练参数
net = Net()
optimizer = torch.optim.Adam(net.parameters(), lr=LR)
loss_func = nn.MSELoss()

for epoch in range(EPOCH):
    '''
    xList = []  # 记录绘图数据
    yList = []
    zList = []
    '''
    for step, x in enumerate(data_loader):
        b_x = x.view(-1, IMG_H * IMG_W)
        b_y = x.view(-1, IMG_H * IMG_W)

        enZ, decoded = net(b_x)

        loss = loss_func(decoded, b_y)  # mean square error
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        print("EPOCH:", epoch + 1, "LOSS:", loss.data)

        '''
        # x, y, z 的数据值
        X = enZ.data[:, 0].numpy()
        Y = enZ.data[:, 1].numpy()
        Z = enZ.data[:, 2].numpy()
        xList.append(X)
        yList.append(Y)
        zList.append(Z)
        '''
    '''
    # 制图
    fig = plt.figure()
    ax = Axes3D(fig)  # 3D 图
    xList=np.array(xList)
    yList=np.array(yList)
    zList=np.array(zList)
    ax.plot_surface(xList, yList, zList, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
    plt.show()
    '''


torch.save(net, path + 'save/net.pkl')
