# Author:  Acer Zhang
# Datetime:2019/10/11
# Copyright belongs to the author.
# Please indicate the source for reprinting.
from __future__ import division
from __future__ import print_function

import sys
import os
import matplotlib
import PIL
import six
import numpy as np
import math
import time
import paddle
import paddle.fluid as fluid

matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot(gen_data):
    pad_dim = 1
    paded = pad_dim + img_dim
    gen_data = gen_data.reshape(gen_data.shape[0], img_dim, img_dim)
    n = int(math.ceil(math.sqrt(gen_data.shape[0])))
    gen_data = (np.pad(
        gen_data, [[0, n * n - gen_data.shape[0]], [pad_dim, 0], [pad_dim, 0]],
        'constant').reshape((n, n, paded, paded)).transpose((0, 2, 1, 3))
                .reshape((n * paded, n * paded)))
    fig = plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(gen_data, cmap='Greys_r', vmin=-1, vmax=1)
    return fig


gf_dim = 64  # 生成器的feature map的基础通道数量，生成器中所有的feature map的通道数量都是基础通道数量的倍数
df_dim = 64  # 判别器的feature map的基础通道数量，判别器中所有的feature map的通道数量都是基础通道数量的倍数
gfc_dim = 1024 * 2  # 生成器的全连接层维度
dfc_dim = 1024  # 判别器的全连接层维度
img_dim = 28  # 输入图片的尺寸

NOISE_SIZE = 100  # 输入噪声的维度
LEARNING_RATE = 2e-4  # 训练的学习率

epoch = 200  # 训练的epoch数
output = "./output_dcgan"  # 模型和测试结果的存储路径
use_cudnn = False  # 是否使用cuDNN
use_gpu = True  # 是否使用GPU训练


def bn(x, name=None, act='relu'):
    return fluid.layers.batch_norm(
        x,
        param_attr=name + '1',
        bias_attr=name + '2',
        moving_mean_name=name + '3',
        moving_variance_name=name + '4',
        name=name,
        act=act)


def conv(x, num_filters, name=None, act=None):
    return fluid.nets.simple_img_conv_pool(
        input=x,
        filter_size=5,
        num_filters=num_filters,
        pool_size=2,
        pool_stride=2,
        param_attr=name + 'w',
        bias_attr=name + 'b',
        use_cudnn=use_cudnn,
        act=act)


def fc(x, num_filters, name=None, act=None):
    return fluid.layers.fc(input=x,
                           size=num_filters,
                           act=act,
                           param_attr=name + 'w',
                           bias_attr=name + 'b')


def deconv(x,
           num_filters,
           name=None,
           filter_size=5,
           stride=2,
           dilation=1,
           padding=2,
           output_size=None,
           act=None):
    return fluid.layers.conv2d_transpose(
        input=x,
        param_attr=name + 'w',
        bias_attr=name + 'b',
        num_filters=num_filters,
        output_size=output_size,
        filter_size=filter_size,
        stride=stride,
        dilation=dilation,
        padding=padding,
        use_cudnn=use_cudnn,
        act=act)


def D(x):
    x = fluid.layers.reshape(x=x, shape=[-1, 1, 28, 28])
    x = conv(x, df_dim, act='leaky_relu', name='conv1')
    x = bn(conv(x, df_dim * 2, name='conv2'), act='leaky_relu', name='bn1')
    x = bn(fc(x, dfc_dim, name='fc1'), act='leaky_relu', name='bn2')
    x = fc(x, 1, act='sigmoid', name='fc2')
    return x


def G(x):
    x = bn(fc(x, gfc_dim, name='fc3'), name='bn3')
    x = bn(fc(x, gf_dim * 2 * img_dim // 4 * img_dim // 4, name='fc4'), name='bn4')
    x = fluid.layers.reshape(x, [-1, gf_dim * 2, img_dim // 4, img_dim // 4])
    x = deconv(x, gf_dim * 2, act='relu', output_size=[14, 14], name='deconv1')
    x = deconv(x, num_filters=1, filter_size=5, padding=2, act='tanh', output_size=[28, 28], name='deconv2')
    x = fluid.layers.reshape(x, shape=[-1, 28 * 28])
    return x


def loss(x, label):
    return fluid.layers.mean(
        fluid.layers.sigmoid_cross_entropy_with_logits(x=x, label=label))


d_program = fluid.Program()
dg_program = fluid.Program()

# 定义判别真实图片的program
with fluid.program_guard(d_program):
    # 输入图片大小为28*28=784
    img = fluid.layers.data(name='img', shape=[784], dtype='float32')
    # 标签shape=1
    label = fluid.layers.data(name='label', shape=[1], dtype='float32')
    d_logit = D(img)
    d_loss = loss(d_logit, label)

# 定义判别生成图片的program
with fluid.program_guard(dg_program):
    noise = fluid.layers.data(
        name='noise', shape=[NOISE_SIZE], dtype='float32')
    # 噪声数据作为输入得到生成图片
    g_img = G(x=noise)

    g_program = dg_program.clone()
    g_program_test = dg_program.clone(for_test=True)

    # 判断生成图片为真实样本的概率
    dg_logit = D(g_img)

    # 计算生成图片被判别为真实样本的loss
    dg_loss = loss(
        dg_logit,
        fluid.layers.fill_constant_batch_size_like(
            input=noise, dtype='float32', shape=[-1, 1], value=1.0))

opt = fluid.optimizer.Adam(learning_rate=LEARNING_RATE)
opt.minimize(loss=d_loss)
parameters = [p.name for p in g_program.global_block().all_parameters()]
opt.minimize(loss=dg_loss, parameter_list=parameters)

batch_size = 128  # Minibatch size

train_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.mnist.train(), buf_size=60000),
    batch_size=batch_size)

if use_gpu:
    exe = fluid.Executor(fluid.CUDAPlace(0))
else:
    exe = fluid.Executor(fluid.CPUPlace())

exe.run(fluid.default_startup_program())

t_time = 0
losses = [[], []]

# 判别器的迭代次数
NUM_TRAIN_TIMES_OF_DG = 2

# 最终生成图像的噪声数据
const_n = np.random.uniform(
    low=-1.0, high=1.0,
    size=[batch_size, NOISE_SIZE]).astype('float32')

for pass_id in range(epoch):
    for batch_id, data in enumerate(train_reader()):
        if len(data) != batch_size:
            continue

        # 生成训练过程的噪声数据
        noise_data = np.random.uniform(
            low=-1.0, high=1.0,
            size=[batch_size, NOISE_SIZE]).astype('float32')

        # 真实图片
        real_image = np.array(list(map(lambda x: x[0], data))).reshape(
            -1, 784).astype('float32')
        # 真实标签
        real_labels = np.ones(
            shape=[real_image.shape[0], 1], dtype='float32')
        # 虚假标签
        fake_labels = np.zeros(
            shape=[real_image.shape[0], 1], dtype='float32')
        total_label = np.concatenate([real_labels, fake_labels])
        s_time = time.time()

        # 虚假图片
        generated_image = exe.run(g_program,
                                  feed={'noise': noise_data},
                                  fetch_list={g_img})[0]
        fluid.io.save_persistables(executor=exe, dirname="./g_program" + str(epoch), main_program=g_program)
        total_images = np.concatenate([real_image, generated_image])

        # D 判断虚假图片为假的loss
        d_loss_1 = exe.run(d_program,
                           feed={
                               'img': generated_image,
                               'label': fake_labels,
                           },
                           fetch_list={d_loss})[0][0]

        # D 判断真实图片为真的loss
        d_loss_2 = exe.run(d_program,
                           feed={
                               'img': real_image,
                               'label': real_labels,
                           },
                           fetch_list={d_loss})[0][0]
        fluid.io.save_persistables(executor=exe, dirname="./d_program" + str(epoch), main_program=d_program)
        d_loss_n = d_loss_1 + d_loss_2
        losses[0].append(d_loss_n)

        # 训练生成器
        for _ in six.moves.xrange(NUM_TRAIN_TIMES_OF_DG):
            noise_data = np.random.uniform(
                low=-1.0, high=1.0,
                size=[batch_size, NOISE_SIZE]).astype('float32')
            dg_loss_n = exe.run(dg_program,
                                feed={'noise': noise_data},
                                fetch_list={dg_loss})[0][0]
            fluid.io.save_persistables(executor=exe, dirname="./dg_program" + str(epoch), main_program=dg_program)
            losses[1].append(dg_loss_n)
        t_time += (time.time() - s_time)
        if batch_id % 10 == 0:
            if not os.path.exists(output):
                os.makedirs(output)
            # 每轮的生成结果
            generated_images = exe.run(g_program_test,
                                       feed={'noise': const_n},
                                       fetch_list={g_img})[0]
            # 将真实图片和生成图片连接
            total_images = np.concatenate([real_image, generated_images])
            fig = plot(total_images)
            msg = "Epoch ID={0} Batch ID={1} D-Loss={2} DG-Loss={3}\n ".format(
                pass_id, batch_id,
                d_loss_n, dg_loss_n)
            print(msg)
            plt.title(msg)
            plt.savefig(
                '{}/{:04d}_{:04d}.png'.format(output, pass_id,
                                              batch_id),
                bbox_inches='tight')
            plt.close(fig)


def display_image(epoch_no, batch_id):
    return PIL.Image.open('output_dcgan/{:04d}_{:04d}.png'.format(epoch_no, batch_id))


# 观察第10个epoch，460个batch的生成图像：
display_image(10, 460)
