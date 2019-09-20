# Author:  Acer Zhang
# Datetime:2019/9/14
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import paddle.fluid as fluid


# img_size = (512, 512)

def build_backbone_net(ipt):
    """
    骨干网络
    :param ipt: 输入数据
    :return: 网络输出
    """

    def conv_p_bn(ipt_layer, name_id, filter_size=3, num_filters=32, padding=1):
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
        tmp = fluid.layers.batch_norm(input=tmp, name='bn1')
        return tmp

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


def make_block(ipt):
    """
    对网络返回数据进行切块
    :param ipt: 骨干网络所返回的数据
    :return: 切块列表
    """
    backbone_net = build_backbone_net(ipt)
    conv_block_list = []

    for x_id in range(backbone_net.shape[-1]):
        for y_id in range(backbone_net.shape[-2]):
            block = fluid.layers.slice(backbone_net,
                                       axes=[2, 3],
                                       starts=[y_id, x_id],
                                       ends=[y_id + 1, x_id + 1])
            conv_block_list.append(block)
    # print(backbone_net.shape)
    # print(conv_block_list[0].shape)
    return conv_block_list


class BGSODNet:
    def __init__(self, class_dim=10):
        self.fc_size = 5 + class_dim

    def net(self, img_ipt, box_ipt_list, label_list):
        conv_block_list = make_block(img_ipt)
        tmp_copy = None
        sum_loss = None
        for i, block in enumerate(conv_block_list):
            out_info = self.__base_net(block)
            if i == 0:
                tmp_copy = out_info

            # 获取监督学习标签
            box_ipt = fluid.layers.slice(box_ipt_list, [1], [i], [i + 1])
            label = fluid.layers.slice(label_list, [1], [i], [i + 1])
            label = fluid.layers.reshape(label, shape=[-1, 1])
            loss = self.__cal_loss(out_info, box_ipt, label)
            sum_loss = 2*loss
            tmp_copy = fluid.layers.concat([tmp_copy, out_info])
            # print(sum_loss.shape)
            # print(loss.shape)
            # print(out_info.shape)
            # print(tmp_copy.shape)
            # print("Build Net END")

        return tmp_copy, sum_loss

    def __base_net(self, ipt):
        """
        对结果进行分割
        :param ipt: 已经分割好的小块数据
        :return: 是否有目标的概率，物体坐标数据，标签预测数据
        """
        tmp = fluid.layers.conv2d(input=ipt,
                                  num_filters=64,
                                  filter_size=1,
                                  padding=0,
                                  stride=1,
                                  name='base_conv1')

        tmp = fluid.layers.fc(tmp, size=self.fc_size)
        # 可以不再这里进行切片，在计算loss的时候将进行切片，这样裤省去一步合并的操作
        # pc = fluid.layers.slice(tmp, axes=[1], starts=[0], ends=[1])
        # box = fluid.layers.slice(tmp, axes=[1], starts=[1], ends=[5])
        # forecast = fluid.layers.slice(tmp, axes=[1], starts=[5], ends=[self.fc_size])
        # forecast = fluid.layers.softmax(forecast)
        # return pc, box, forecast
        return tmp

    def __cal_loss(self, fc_out, box_ipt, label):
        """

        :param fc_out: 包含是否有目标的概率，物体坐标数据，标签预测的数据
        :param box_ipt:监督学习box数据
        :param label:监督学习label数据
        :return:
        """
        pc = fluid.layers.slice(fc_out, axes=[1], starts=[0], ends=[1])
        box = fluid.layers.slice(fc_out, axes=[1], starts=[1], ends=[5])
        forecast = fluid.layers.slice(fc_out, axes=[1], starts=[5], ends=[self.fc_size])
        forecast = fluid.layers.softmax(forecast)
        pc_loss = fluid.layers.square_error_cost(pc, fluid.layers.ones(shape=[1], dtype='float32'))
        # box_loss = fluid.layers.square_error_cost(box_ipt, box)
        # avg_box_loss = fluid.layers.mean(box_loss)
        # label_loss = fluid.layers.cross_entropy(forecast, label)
        # avg_label_loss = fluid.layers.mean(label_loss)
        # loss = fluid.layers.elementwise_mul(pc_loss, avg_box_loss)
        # loss = fluid.layers.elementwise_add(loss, avg_label_loss)
        return fluid.layers.reduce_mean(pc_loss)

# Test
# a = fluid.layers.data(name="a", shape=[3, 512, 512], dtype="float32")
# box = fluid.layers.data(name="box", shape=[-1, 4], dtype="float32")
# label = fluid.layers.data(name="label", shape=[-1, 1], dtype="int64")
# result_list, loss = BGSODNet(10).net(a, box, label)
