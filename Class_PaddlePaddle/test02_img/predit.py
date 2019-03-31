# 加载库
import paddle.fluid as fluid
import numpy as np
import Class_OS.o1_获得当前工作目录
from PIL import Image
import numpy
# 指定路径
path = Class_OS.o1_获得当前工作目录.main()
params_dirname = path + "test02.inference.model"
print("训练后文件夹路径" + params_dirname)
# 目标数据
with open(path + "data/ocrData.txt", 'rt') as f:
    a = f.read()


def dataReader(i):
    im = Image.open(path + "data/" + str(i) + ".jpg").convert('L')
    im = numpy.array(im).reshape(1, 1, 30, 15).astype(numpy.float32)
    im = im / 255.0 * 2.0 - 1.0
    #im = numpy.expand_dims(im, axis=0)
    return im


# 参数初始化
cpu = fluid.CPUPlace()
exe = fluid.Executor(cpu)
prog = fluid.default_startup_program()
exe.run(prog)

# 加载模型
[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(params_dirname, exe)

img=dataReader(1972)
results = exe.run(inference_program,
    feed={feed_target_names[0]: img},
    fetch_list=fetch_targets)

lab = np.argsort(results)[0][0][-1]

print(lab)