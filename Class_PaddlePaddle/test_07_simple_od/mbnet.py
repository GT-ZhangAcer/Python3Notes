# Author:  Acer Zhang
# Datetime:2019/10/6
import paddle
import paddle.fluid as fluid

from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr


class MobileNetSSD:
    def __init__(self):
        pass

    def conv_bn(self,
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
            act=None,
            use_cudnn=use_cudnn,
            param_attr=parameter_attr,
            bias_attr=False)
        return fluid.layers.batch_norm(input=conv, act=act)

    def depthwise_separable(self, input, num_filters1, num_filters2, num_groups, stride, scale):
        depthwise_conv = self.conv_bn(
            input=input,
            filter_size=3,
            num_filters=int(num_filters1 * scale),
            stride=stride,
            padding=1,
            num_groups=int(num_groups * scale),
            use_cudnn=False)

        pointwise_conv = self.conv_bn(
            input=depthwise_conv,
            filter_size=1,
            num_filters=int(num_filters2 * scale),
            stride=1,
            padding=0)
        return pointwise_conv

    def extra_block(self, input, num_filters1, num_filters2, num_groups, stride, scale):
        # 1x1 conv
        pointwise_conv = self.conv_bn(
            input=input,
            filter_size=1,
            num_filters=int(num_filters1 * scale),
            stride=1,
            num_groups=int(num_groups * scale),
            padding=0)

        # 3x3 conv
        normal_conv = self.conv_bn(
            input=pointwise_conv,
            filter_size=3,
            num_filters=int(num_filters2 * scale),
            stride=2,
            num_groups=int(num_groups * scale),
            padding=1)
        return normal_conv

    def net(self, img, box_ipt_list, label_list, scale=1.0):
        # 300x300
        tmp = self.conv_bn(img, 3, int(32 * scale), 2, 1)
        # 150x150
        tmp = self.depthwise_separable(tmp, 32, 64, 32, 1, scale)
        tmp = self.depthwise_separable(tmp, 64, 128, 64, 2, scale)
        # 75x75
        tmp = self.depthwise_separable(tmp, 128, 128, 128, 1, scale)
        tmp = self.depthwise_separable(tmp, 128, 256, 128, 2, scale)
        # 38x38
        tmp = self.depthwise_separable(tmp, 256, 256, 256, 1, scale)
        tmp = self.depthwise_separable(tmp, 256, 512, 256, 2, scale)

        # 19x19
        for i in range(5):
            tmp = self.depthwise_separable(tmp, 512, 512, 512, 1, scale)
        module11 = tmp
        tmp = self.depthwise_separable(tmp, 512, 1024, 512, 2, scale)

        # 10x10
        module13 = self.depthwise_separable(tmp, 1024, 1024, 1024, 1, scale)
        module14 = self.extra_block(module13, 256, 512, 1, 2, scale)
        # 5x5
        module15 = self.extra_block(module14, 128, 256, 1, 2, scale)
        # 3x3
        module16 = self.extra_block(module15, 128, 256, 1, 2, scale)
        # 2x2
        module17 = self.extra_block(module16, 64, 128, 1, 2, scale)

        mbox_locs, mbox_confs, boxs, vars = fluid.layers.multi_box_head(
            inputs=[module11, module13, module14, module15, module16, module17],
            image=img,
            num_classes=3,
            min_ratio=20,
            max_ratio=90,
            aspect_ratios=[[2.], [2., 3.], [2., 3.], [2., 3.], [2., 3.], [2., 3.]],
            base_size=300,
            offset=0.5,
            flip=True,
            clip=True)

        loss = fluid.layers.ssd_loss(location=mbox_locs,
                                     confidence=mbox_confs,
                                     gt_box=box_ipt_list,
                                     gt_label=label_list,
                                     prior_box=boxs,
                                     prior_box_var=vars)
        loss = fluid.layers.mean(loss)
        nmsed_out = fluid.layers.detection_output(mbox_locs, mbox_confs, boxs, vars, nms_threshold=0.45)  # 非极大值抑制得到的结果
        map_eval = fluid.metrics.DetectionMAP(nmsed_out, label_list, box_ipt_list, class_num=3, overlap_threshold=0.5,
                                              evaluate_difficult=False, ap_version='11point')
        cur_map, accum_map = map_eval.get_map_var()
        return loss, cur_map, accum_map
