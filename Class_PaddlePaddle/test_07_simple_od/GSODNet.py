# Author:  Acer Zhang
# Datetime:2019/9/14
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay


# img_size = (512, 512)

def conv_p_bn(ipt_layer, name_id, filter_size=3, num_filters=32, padding=1):
    """
    卷积池化BN层三合一函数
    :param ipt_layer: 输入层
    :param name_id: 层标识名
    :param filter_size: 卷积核大小
    :param num_filters: 卷积核个数
    :param padding: 填充
    :return: [N x C x H x W] N: Batch_size, C: channel, H: layer_H, W: Layer_W
    """
    tmp = fluid.layers.conv2d(input=ipt_layer,
                              num_filters=num_filters,
                              filter_size=filter_size,
                              padding=padding,
                              stride=1,
                              name='conv' + str(name_id),
                              act='relu')
    tmp = fluid.layers.pool2d(input=tmp,
                              pool_size=2,
                              pool_stride=2,
                              pool_type='max',
                              name='pool' + str(name_id))
    out = fluid.layers.batch_norm(
        input=tmp, act='relu',
        param_attr=ParamAttr(initializer=fluid.initializer.Normal(0., 0.02), regularizer=L2Decay(0.)),
        bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), regularizer=L2Decay(0.)))
    return out


def build_backbone_net(ipt):
    """
    骨干网络
    :param ipt: 输入数据
    :return: 网络输出
    """
    layer_1 = conv_p_bn(ipt, 1, filter_size=3, padding=1)
    layer_2 = conv_p_bn(layer_1, 2, filter_size=3, padding=1, num_filters=64)
    layer_3 = conv_p_bn(layer_2, 3, filter_size=3, padding=1, num_filters=64)
    layer_4 = conv_p_bn(layer_3, 4, filter_size=3, padding=1, num_filters=128)
    layer_5 = conv_p_bn(layer_4, 5, filter_size=3, padding=1, num_filters=128)
    # print(layer_1.shape)
    # print(layer_2.shape)
    # print(layer_3.shape)
    # print(layer_4.shape)
    # print(layer_5.shape)
    # print("Base Net END")
    return layer_5


class BGSODNet:
    def __init__(self, class_dim=10):
        # self.fc_size = [Pc, box_x,y,w,h ,class_dim...,]
        self.fc_size = 5 + class_dim

    def net(self, img_ipt, box_ipt_list, label_list):
        anchors = [7, 14, 15, 21, 30, 45]

        layer_out = build_backbone_net(img_ipt)
        layer_out = self.__make_net(layer_out)
        img_size = fluid.layers.data(name='img_size', shape=[2], dtype='int64')
        boxes, scores = fluid.layers.yolo_box(x=layer_out,
                                              img_size=img_size,
                                              class_num=self.fc_size - 5,
                                              anchors=anchors,
                                              conf_thresh=0.01,
                                              downsample_ratio=32)
        loss = fluid.layers.yolov3_loss(layer_out,
                                        gt_box=box_ipt_list,
                                        gt_label=label_list,  # 必须是int32 坑死了
                                        anchors=anchors,
                                        anchor_mask=[0],  # 取决于合成特征图层个数，此处没有合成
                                        class_num=self.fc_size - 5,
                                        ignore_thresh=0.5,
                                        downsample_ratio=32)
        return boxes, scores, loss

    def __make_net(self, backbone_net_out):
        # 1x1卷积降维
        out = conv_p_bn(backbone_net_out, name_id=10, filter_size=1, num_filters=self.fc_size)
        return out


# Test
# a = fluid.layers.data(name="a", shape=[3, 512, 512], dtype="float32")
# box = fluid.layers.data(name="box", shape=[16 * 16, 4], dtype="float32")
# label = fluid.layers.data(name="label", shape=[16 * 16], dtype="int64")
# BGSODNet(10).net(a, box, label)
