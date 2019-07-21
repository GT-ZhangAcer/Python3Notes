import paddle.fluid as fluid
import numpy as np

x = fluid.layers.data(name="x", shape=[1], dtype="int64")
y = fluid.layers.data(name="y", shape=[1], dtype="int64")
f = fluid.layers.sum(x=[x, y])

cpu = fluid.CUDAPlace(0)  # 此处使用CPU进行训练 GPU训练则移步之后更新的文章
exe = fluid.Executor(cpu)  # Executor是执行器
prog = fluid.default_startup_program()  # 将刚刚定义的一堆堆赋值给prog这个变量名
exe.run(prog)  # 准备开始！

out = exe.run(feed={"x": np.array([3, 2]), "y": np.array([1, 1])}, fetch_list=[x, y, f])

for i in out:
    print(i)
