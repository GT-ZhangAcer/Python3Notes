# Author:  Acer Zhang
# Datetime:2019/9/23
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import paddle.fluid as fluid
from GSODNet import BGSODNet

# Hyper parameter
use_cuda = True  # Whether to use GPU or not
save_model_path = "./model"
img_size = [512, 512]
block_num = 16  # 分块个数

# Initialization
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)

# Program
main_program = fluid.Program()
startup = fluid.Program()

# Edit Program
with fluid.program_guard(main_program=main_program, startup_program=startup):
    """Tips:Symbol * stands for Must"""
    # * Define data types
    img = fluid.layers.data(name="img", shape=[3, img_size[0], img_size[1]], dtype="float32")
    box = fluid.layers.data(name="box", shape=[block_num ** 2, 4], dtype="float32")
    label = fluid.layers.data(name="label", shape=[block_num ** 2], dtype="int32")
    img_size_2d = fluid.layers.data(name='img_size', shape=[2], dtype='int32')
    # * Access to the Network
    scores, boxes = BGSODNet(10).net(img, box, label, img_size_2d, for_train=False)

# Train Process
exe.run(startup)

fluid.io.load_params(executor=exe, dirname=save_model_path + "/One_Epoch", main_program=main_program)
fluid.io.save_inference_model(dirname=save_model_path + "/infer", feeded_var_names=["img", "img_size"],
                              target_vars=[boxes],
                              executor=exe, main_program=main_program)
