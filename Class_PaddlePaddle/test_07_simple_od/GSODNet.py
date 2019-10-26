# Author:  Acer Zhang
# Datetime:2019/9/14
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import MSRA
from resnet_simple import SimpleResNet


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
    conv = fluid.layers.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        stride=stride,
        padding=padding,
        groups=num_groups,
        use_cudnn=use_cudnn)
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
    # layer_1 = conv_bn(ipt, filter_size=3, num_filters=64, stride=2, padding=1)
    # layer_2 = conv_bn(layer_1, filter_size=3, num_filters=128, stride=2, padding=1)
    # layer_3 = conv_bn(layer_2, filter_size=3, num_filters=128, stride=2, padding=1)
    # layer_4 = conv_bn(layer_3, filter_size=3, num_filters=128, stride=2, padding=1)
    # layer_5 = conv_bn(layer_4, filter_size=3, num_filters=256, stride=2, padding=1)
    # layer_6 = conv_bn(layer_5, filter_size=3, num_filters=256, stride=2, padding=1)
    # layer_7 = conv_bn(layer_6, filter_size=3, num_filters=512, stride=2, padding=1)
    # layer_8 = conv_bn(layer_7, filter_size=3, num_filters=512, stride=2, padding=1)
    # # print(layer_2.shape, layer_3.shape, layer_4.shape)
    # return [layer_4, layer_6, layer_8]
    net_obj = SimpleResNet(ipt)
    return net_obj.req_detection_net()


class BGSODNet:
    def __init__(self, class_dim=10):
        self.fc_size = 5 + class_dim

    def net(self, img_ipt, box_ipt_list, label_list, for_test=False):

        layer_out = build_backbone_net(img_ipt)

        mbox_locs, mbox_confs, boxs, vars = fluid.layers.multi_box_head(
            inputs=layer_out,
            image=img_ipt,
            num_classes=3,
            min_ratio=3,
            max_ratio=50,
            aspect_ratios=[[1.], [1., 2.], [1., 3.]],
            base_size=300,
            offset=0.5,
            flip=True,
            clip=True)

        nms_out = fluid.layers.detection_output(mbox_locs, mbox_confs, boxs, vars, nms_threshold=0.45)  # 非极大值抑制得到的结果
        if for_test:
            return nms_out
        else:
            loss = fluid.layers.ssd_loss(location=mbox_locs,
                                         confidence=mbox_confs,
                                         gt_box=box_ipt_list,
                                         gt_label=label_list,
                                         prior_box=boxs,
                                         prior_box_var=vars,
                                         background_label=0)
            loss = fluid.layers.mean(loss)
            map_eval = fluid.metrics.DetectionMAP(nms_out, label_list, box_ipt_list, class_num=3,
                                                  overlap_threshold=0.5,
                                                  evaluate_difficult=False, ap_version='11point')
            cur_map, accum_map = map_eval.get_map_var()
            return loss, cur_map, accum_map, map_eval

# Test
# a = fluid.layers.data(name="a", shape=[3, 512, 512], dtype="float32")
# box = fluid.layers.data(name="box", shape=[16 * 16, 4], dtype="float32")
# label = fluid.layers.data(name="label", shape=[16 * 16], dtype="int64")
# BGSODNet(10).net(a, box, label)
