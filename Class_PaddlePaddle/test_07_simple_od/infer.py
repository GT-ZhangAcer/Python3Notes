# Author:  Acer Zhang
# Datetime:2019/9/23
# Copyright belongs to the author.
# Please indicate the source for reprinting.
import time
import paddle.fluid as fluid
import paddle
import numpy as np
from PIL import Image
from PIL import ImageDraw

# Hyper parameter
use_cuda = False  # Whether to use GPU or not
batch_size = 100  # Number of incoming batches of data
model_path = "./model/infer"  # infer model path
data_path = "./lslm_data"


# index = 18


def data_reader():
    def reader():
        for index in range(11, 41):
            im = Image.open(data_path + "/" + str(index) + ".jpg")
            im = im.crop((360, 0, 1440, 1080))
            im = im.resize((300, 300), Image.LANCZOS)
            im = np.array(im).transpose((2, 0, 1)).reshape(1, 3, 300, 300) * 0.007843
            yield im

    return reader


# Initialization

place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)
startup = fluid.Program()

# load infer model
[infer_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(model_path, exe)


def draw_bbox_image(img, nms_out):
    confs_threshold = 0.7  # 置信度
    draw = ImageDraw.Draw(img)
    for dt in nms_out:
        if dt[1] < confs_threshold:
            continue
        this_label = dt[0]
        bbox = dt[2:]
        # 根据网络输出，获取矩形框的左上角、右下角坐标相对位置
        draw.rectangle((360 + bbox[0] * 1080, bbox[1] * 1080, bbox[2] * 1080, bbox[3] * 1080), None, 'red')
    img.show()


exe.run(startup)
# Start infer
infer_reader = paddle.batch(reader=data_reader(), batch_size=batch_size)
infer_feeder = fluid.DataFeeder(place=place, feed_list=feed_target_names, program=infer_program)

start_time = time.time()
for data in infer_reader():
    results = exe.run(infer_program, feed=infer_feeder.feed(data),
                      fetch_list=fetch_targets, return_numpy=False)
    # nms_out = np.asarray(results[0])
    # im = Image.open(data_path + "/" + str(index) + ".jpg")
    # draw_bbox_image(im, nms_out)
    # print(nms_out)
print(time.time() - start_time)  # [1]4.6 [10]4.5 [40]4.5 [100]41
