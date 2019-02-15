#加载库
import paddle.fluid as fluid
import numpy
import Class_OS.o1_获得当前工作目录

#指定路径
path=Class_OS.o1_获得当前工作目录.main()+"/"
params_dirname = path+"test01.inference.model"
print("训练后文件夹路径"+params_dirname)
#目标数据
datatype="float32"
test_data=numpy.array([[input("请输入数值")]]).astype(datatype)#测试数为60

#参数初始化
cpu = fluid.CPUPlace()
exe = fluid.Executor(cpu)
prog=fluid.default_startup_program()

# 加载模型
[inference_program, feed_target_names,fetch_targets] = fluid.io.load_inference_model(params_dirname, exe)
#读取训练后文件
fluid.io.load_params(executor=exe, dirname=params_dirname,
                     main_program=prog)

results = exe.run(inference_program,
                  feed={feed_target_names[0]: test_data},
                  fetch_list=fetch_targets)
print(results[0][0])
print(fetch_targets)
print(feed_target_names)#网格列表