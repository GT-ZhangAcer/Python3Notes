# Author:  Acer Zhang
# Datetime:2019/9/13
# Copyright belongs to the author.
# Please indicate the source for reprinting.
import PIL.Image as Image
import numpy as np
import paddle.fluid as fluid

exe = fluid.Executor(fluid.CPUPlace())
fluid.io.load_params(exe, r"F:\Python3Notes\Class_PaddlePaddle\test_07_simple_od\xception65_bn_cityscapes")
