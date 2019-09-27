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
                              act='relu',
                              param_attr=ParamAttr(initializer=fluid.initializer.Normal(0., 0.02)),
                              bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), regularizer=L2Decay(0.)))
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


def down_pool(ipt_layer, name: int):
    tmp = fluid.layers.pool2d(input=ipt_layer,
                              pool_size=2,
                              pool_stride=2,
                              pool_type='max',
                              name='down_pool' + str(name))
    return tmp


def build_backbone_net(ipt):
    """
    骨干网络
    :param ipt: 输入数据
    :return: 网络输出
    """
    # layer_1 = conv_p_bn(ipt, 1, filter_size=3, padding=0)
    # layer_2 = conv_p_bn(layer_1, 2, filter_size=3, padding=0, num_filters=64)
    # layer_3 = conv_p_bn(layer_2, 3, filter_size=3, padding=0, num_filters=64)
    # layer_4 = conv_p_bn(layer_3, 4, filter_size=3, padding=0, num_filters=128)
    # layer_5 = conv_p_bn(layer_4, 5, filter_size=3, padding=0, num_filters=128)
    layer_1 = conv_p_bn(ipt, 1, filter_size=3, padding=1)
    layer_2 = conv_p_bn(layer_1, 2, filter_size=3, padding=1, num_filters=64)
    layer_3 = conv_p_bn(layer_2, 3, filter_size=3, padding=1, num_filters=64)
    layer_4 = conv_p_bn(layer_3, 4, filter_size=3, padding=1, num_filters=128)
    # layer_5 = conv_p_bn(layer_4, 5, filter_size=3, padding=1, num_filters=128)
    # print(layer_1.shape)
    # print(layer_2.shape)
    # print(layer_3.shape)
    # print(layer_4.shape)
    # print(layer_5.shape)
    # print("Base Net END")
    # return layer_5, layer_4, layer_3
    return layer_4


class BGSODNet:
    def __init__(self, class_dim=10):
        # self.fc_size = [Pc, box_x,y,w,h ,class_dim...,]
        self.fc_size = 5 + class_dim

    def net(self, img_ipt, box_ipt_list, label_list, img_size, true_scores=None, for_train=True):
        anchors = [14, 30, 45, 51, 60, 75]

        layer_out = build_backbone_net(img_ipt)
        layer_out = self.__make_net_simple(layer_out)

        print(layer_out.shape)
        boxes, scores = fluid.layers.yolo_box(x=layer_out,
                                              img_size=img_size,
                                              class_num=self.fc_size - 5,
                                              anchors=anchors[:2],
                                              conf_thresh=0.01,
                                              downsample_ratio=32)
        print(boxes.shape, scores.shape)
        scores = fluid.layers.transpose(scores, perm=[0, 2, 1])
        if for_train:

            loss = fluid.layers.yolov3_loss(layer_out,
                                            gt_box=box_ipt_list,
                                            gt_label=label_list,  # 必须是int32 坑死了
                                            gt_score=true_scores,
                                            anchors=anchors,
                                            # anchor_mask=[0, 1, 2],  # 取决于合成特征图层个数，此处没有合成
                                            anchor_mask=[0],
                                            class_num=self.fc_size - 5,
                                            ignore_thresh=0.1,
                                            downsample_ratio=32)

            # -----
            # scores = fluid.layers.transpose(scores, [0, 2, 1])
            #
            # scores_loss = fluid.layers.cross_entropy(scores, label_list)
            # scores_loss = fluid.layers.mean(scores_loss)
            # loss = scores_loss
            # -----


            # loss = fluid.layers.elementwise_add(scores)
            return scores, loss
        else:
            out_box = fluid.layers.multiclass_nms(bboxes=boxes,
                                                  scores=scores,
                                                  background_label=10,
                                                  score_threshold=0.25,
                                                  nms_top_k=400,
                                                  nms_threshold=0.3,
                                                  keep_top_k=-1)
            return scores, out_box

    def __make_net(self, backbone_net_out):
        # 1x1卷积降维,对大尺度特征图进行池化，使其尺寸一致
        layers = []
        for i, out_layer in enumerate(backbone_net_out):
            for num in range(i):
                out_layer = down_pool(out_layer, i * 10 + num)
            out = conv_p_bn(out_layer, name_id=10 + i, filter_size=1, num_filters=self.fc_size)
            layers.append(out)
        layers_out = fluid.layers.concat(layers, axis=1)
        return layers_out

    def __make_net_simple(self, backbone_net_out):
        # 要求输入不能是列表
        out = fluid.layers.conv2d(input=backbone_net_out,
                                  num_filters=self.fc_size,
                                  filter_size=1,
                                  padding=0,
                                  stride=1,
                                  name="end_layer",
                                  param_attr=ParamAttr(initializer=fluid.initializer.Normal(0., 0.02)),
                                  bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0),
                                                      regularizer=L2Decay(0.)))

        return out

# Test
# a = fluid.layers.data(name="a", shape=[3, 512, 512], dtype="float32")
# box = fluid.layers.data(name="box", shape=[16 * 16, 4], dtype="float32")
# label = fluid.layers.data(name="label", shape=[16 * 16], dtype="int64")
# BGSODNet(10).net(a, box, label)
