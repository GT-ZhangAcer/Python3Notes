import time
import os


class WriteLog:
    def __init__(self, path="./"):
        this_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        self.now_train_loss = None
        self.now_test_loss = None
        self.now_train_acc1 = None
        self.now_train_acc5 = None
        self.now_test_acc1 = None
        self.now_test_acc5 = None
        self.now_step = 0
        self.batch_train_acc1 = []
        self.batch_train_acc5 = []


        with open(path + str(this_time) + ".log", "w") as f:
            f.write(str(this_time)+"/n")
        print("WriteLog is ready !")
    def add_batch_train_acc1(self,acc1):
