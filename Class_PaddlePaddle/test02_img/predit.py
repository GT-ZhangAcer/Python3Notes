#加载库
import paddle.fluid as fluid
import numpy as np
import Class_OS.o1_获得当前工作目录
from PIL import Image

#指定路径
path=Class_OS.o1_获得当前工作目录.main()
params_dirname = path+"test02.inference.model"
print("训练后文件夹路径"+params_dirname)
#目标数据
with open(path + "data/ocrData.txt", 'rt') as f:
    a=f.read()


def load_image(i):
    file=path + "data/%s.jpg"%i
    img = Image.open(file)
    # 统一图像大小
    img = img.resize((15, 30), Image.ANTIALIAS)
    # 转换成numpy值
    img = np.array(img).astype(np.float32)
    # 转换成CHW
    img = img.transpose((2, 0, 1))
    # 转换成BGR
    img = img[(2, 1, 0), :, :] / 255.0
    img = np.expand_dims(img, axis=0)
    return img


#参数初始化
cpu = fluid.CPUPlace()
exe = fluid.Executor(cpu)
prog=fluid.default_startup_program()

# 加载模型
[inference_program, feed_target_names,fetch_targets] = fluid.io.load_inference_model(params_dirname, exe)



for i in range(1999,2000):
    img=load_image(i)
    results = exe.run(inference_program,
                      feed={feed_target_names[0]: img},
                      fetch_list=fetch_targets)
    print(results[1])
