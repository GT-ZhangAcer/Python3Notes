import time
import os
import numpy as np
from script.os_Script import mkdir


def add_check(uncheck_list, value):
    """
    检查是否有Nan情况，如果有则添加上一组数据至列表中保存
    :param uncheck_list: 未检查的列表
    :param value: 待添加的值
    """
    value = value.tolist()[0]
    value = round(value, 5)
    if type(value) is (float or int):
        uncheck_list.append(value)
    else:
        uncheck_list.append(uncheck_list[-1])


class WriteLog:
    """
    日志写入类
    在训练循环外创建该类的实例
    训练过程中使用add_batch_xxx_value方法来添加训练中每Mini_Batch输出
    训练循环内使用write_and_req方法来获取每Epoch输出，并将该输出写入到文件
    Example:

    log_obj = WriteLog()

    for epoch in range(epochs):
        for step, data in enumerate(batch_reader()):
            outs = exe.run(program=main_program,
                           feed=train_feeder.feed(data),
                           fetch_list=[acc_1, acc_5, loss],
                           return_numpy=False)
            log_obj.add_batch_train_value(outs[1], outs[2], outs[3])

        train_print, _ = log_obj.write_and_req()
        print("Avg acc1 ", train_print["acc1"], "acc5 ", train_print["acc5"], "loss ", train_print["loss"])

        # If you want a minimalist style , Please try the following code !

        print(log_obj.write_and_req()[1]) # Only print test result

    """

    def __init__(self, path="./"):
        """
        :param path: 日志文件保存路径
        """
        this_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        self.path = os.path.join(path, str(this_time))
        self.batch_train_acc1 = [.0]
        self.batch_train_acc5 = [.0]
        self.batch_train_loss = [.0]
        self.batch_test_acc1 = [.0]
        self.batch_test_acc5 = [.0]
        self.batch_test_loss = [.0]

        mkdir(path, de=False)
        with open(self.path + ".log", "w") as f:
            f.write(str(this_time) + "\n")
        print("WriteLog is ready !")

    def add_batch_train_value(self, acc1, acc5=None, loss=None):
        add_check(self.batch_train_acc1, acc1)
        if acc5 is not None:
            add_check(self.batch_train_acc5, acc5)
        if loss is not None:
            add_check(self.batch_train_loss, loss)

    def add_batch_test_value(self, acc1, acc5=None, loss=None):
        add_check(self.batch_test_acc1, acc1)
        if acc5 is not None:
            add_check(self.batch_test_acc5, acc5)
        if loss is not None:
            add_check(self.batch_test_loss, loss)

    def write_and_req(self):
        """
        写入并获取该Epoch的训练信息
        :return: 训练集字典、测试集字典(acc1,acc5,loss)
        """
        now_train_acc1 = sum(self.batch_train_acc1)
        now_train_acc5 = sum(self.batch_train_acc5)
        now_train_loss = sum(self.batch_train_loss)
        now_test_acc1 = sum(self.batch_test_acc1)
        now_test_acc5 = sum(self.batch_test_acc5)
        now_test_loss = sum(self.batch_test_loss)
        self.batch_train_acc1 = [.0]
        self.batch_train_acc5 = [.0]
        self.batch_train_loss = [.0]
        self.batch_test_acc1 = [.0]
        self.batch_test_acc5 = [.0]
        self.batch_test_loss = [.0]

        with open(self.path + ".log", "a") as f:
            f.writelines(str(now_train_acc1) + "-" + str(now_train_acc5) + "-" + str(now_train_loss)
                         + "_" + str(now_test_acc1) + "-" + str(now_test_acc5) + "-" + str(now_test_loss) + "\n")
        train_dict = {"acc1": now_train_acc1, "acc5": now_train_acc5, "loss": now_train_loss}
        test_dict = {"acc1": now_test_acc1, "acc5": now_test_acc5, "loss": now_test_loss}
        return train_dict, test_dict
