# Author:  Acer Zhang
# Datetime:2019/9/14
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import paddle
import paddle.fluid as fluid


# img_size = (512, 512)

def build_backbone_net(ipt):
    def conv_p_bn(ipt_layer, name_id, filter_size=3, num_filters=64, padding=0):
        tmp = fluid.layers.conv2d(input=ipt_layer,
                                  num_filters=num_filters,
                                  filter_size=filter_size,
                                  padding=padding,
                                  stride=2,
                                  name='conv' + str(name_id),
                                  act='relu')
        tmp = fluid.layers.pool2d(input=tmp,
                                  pool_size=2,
                                  pool_stride=2,
                                  pool_type='max',
                                  name='pool' + str(name_id))
        tmp = fluid.layers.batch_norm(input=tmp, name='bn1')
        return tmp

    layer_1 = conv_p_bn(ipt, 1, filter_size=5)
    layer_2 = conv_p_bn(layer_1, 2, filter_size=3, padding=1, num_filters=128)
    layer_3 = fluid.layers.conv2d(input=layer_2,
                                  num_filters=256,
                                  filter_size=5,
                                  padding=0,
                                  stride=3,
                                  name='conv3',
                                  act='relu')
    print(layer_1.shape)
    print(layer_2.shape)
    print(layer_3.shape)
    return layer_3


def make_block(ipt):
    backbone_net = build_backbone_net(ipt)
    conv_block_list = []
    for block_id in range(backbone_net.shape[-1]):
        block = fluid.layers.slice(backbone_net,
                                   axes=[2, 3],
                                   starts=[block_id, block_id],
                                   ends=[block_id + 1, block_id + 1])
        conv_block_list.append(block)
    return conv_block_list


def cal_loss(pc, box, forecast, box_ipt, label):
    pc_loss = fluid.layers.square_error_cost(pc, fluid.layers.ones(shape=[1], dtype='int64'))
    box_loss = fluid.layers.square_error_cost(box_ipt, box)
    avg_box_loss = fluid.layers.mean(box_loss)
    label_loss = fluid.layers.cross_entropy(forecast, label)
    avg_label_loss = fluid.layers.mean(label_loss)
    loss = fluid.layers.elementwise_add(pc_loss, avg_box_loss)
    loss = fluid.layers.elementwise_add(loss, avg_label_loss)
    return loss


class GSODNet:
    def __init__(self, class_dim=10):
        self.fc_size = 5 + class_dim

    def net(self, img_ipt, box_ipt, label):
        conv_block_list = make_block(img_ipt)
        result_list = []
        for block in conv_block_list:
            pc, box, forecast = self.__base_net(block)
            loss = cal_loss(pc, box, forecast, box_ipt, label)
            result_list.append([pc, box, forecast, loss])
        return result_list

    def __base_net(self, ipt):
        tmp = fluid.layers.conv2d(input=ipt,
                                  num_filters=64,
                                  filter_size=1,
                                  padding=0,
                                  stride=1,
                                  name='base_conv1')

        tmp = fluid.layers.fc(tmp, size=self.fc_size)
        pc = fluid.layers.slice(tmp, axes=[1], starts=[0], ends=[1])
        box = fluid.layers.slice(tmp, axes=[1], starts=[1], ends=[5])
        forecast = fluid.layers.slice(tmp, axes=[1], starts=[5], ends=[self.fc_size])
        forecast = fluid.layers.softmax(forecast)
        return pc, box, forecast


a = fluid.layers.data(name="a", shape=[3, 512, 512], dtype="float32")
GSODNet().net(a)
