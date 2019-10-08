# Author:  Acer Zhang
# Datetime:2019/9/14
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import MSRA


# img_size = (512, 512)

def conv_bn(
        input,
        filter_size,
        num_filters,
        stride,
        padding,
        num_groups=1,
        act='relu',
        use_cudnn=True):
    parameter_attr = ParamAttr(learning_rate=0.1, initializer=MSRA())
    conv = fluid.layers.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        stride=stride,
        padding=padding,
        groups=num_groups,
        use_cudnn=use_cudnn,
        param_attr=parameter_attr,
        bias_attr=False)
    return fluid.layers.batch_norm(input=conv, act=act)


def depthwise_separable(input, num_filters1, num_filters2, num_groups, stride, scale):
    depthwise_conv = conv_bn(
        input=input,
        filter_size=3,
        num_filters=int(num_filters1 * scale),
        stride=stride,
        padding=1,
        num_groups=int(num_groups * scale),
        use_cudnn=False)

    pointwise_conv = conv_bn(
        input=depthwise_conv,
        filter_size=1,
        num_filters=int(num_filters2 * scale),
        stride=1,
        padding=0)
    return pointwise_conv


def build_backbone_net(ipt):
    """
    基础网络
    :param ipt: 输入数据
    :return: 网络输出
    """
    layer_1 = conv_bn(ipt, filter_size=3, num_filters=64, stride=2, padding=1)
    layer_2 = conv_bn(layer_1, filter_size=3, num_filters=128, stride=2, padding=1)
    layer_3 = conv_bn(layer_2, filter_size=3, num_filters=256, stride=2, padding=1)
    layer_4 = conv_bn(layer_3, filter_size=3, num_filters=512, stride=2, padding=1)
    print(layer_2.shape, layer_3.shape, layer_4.shape)
    return [layer_2, layer_3, layer_4]


class BGSODNet:
    def __init__(self, class_dim=10):
        self.fc_size = 5 + class_dim

    def net(self, img_ipt, box_ipt_list, label_list, for_train=True):

        layer_out = build_backbone_net(img_ipt)

        mbox_locs, mbox_confs, boxs, vars = fluid.layers.multi_box_head(
            inputs=layer_out,
            image=img_ipt,
            num_classes=10,
            min_ratio=3,
            max_ratio=50,
            aspect_ratios=[[1., 2.], [1., 2.], [1., 2.]],
            base_size=512,
            offset=0.5,
            flip=True,
            clip=True)

        if for_train:

            loss = fluid.layers.ssd_loss(location=mbox_locs,
                                         confidence=mbox_confs,
                                         gt_box=box_ipt_list,
                                         gt_label=label_list,
                                         prior_box=boxs,
                                         prior_box_var=vars)

            return loss
        else:

            scores = fluid.layers.transpose(scores, perm=[0, 2, 1])
            out_box = fluid.layers.multiclass_nms(bboxes=boxes,
                                                  scores=scores,
                                                  background_label=10,
                                                  score_threshold=0.,
                                                  nms_top_k=400,
                                                  nms_threshold=0.3,
                                                  keep_top_k=-1)
            return scores, out_box

# Test
# a = fluid.layers.data(name="a", shape=[3, 512, 512], dtype="float32")
# box = fluid.layers.data(name="box", shape=[16 * 16, 4], dtype="float32")
# label = fluid.layers.data(name="label", shape=[16 * 16], dtype="int64")
# BGSODNet(10).net(a, box, label)
