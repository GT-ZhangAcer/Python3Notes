# Author: Acer Zhang
# Datetime:2019/8/25 17:02
# Copyright belongs to the author.
# Please indicate the source for reprinting.

from script.log_Script import WriteLog  as log

a = log()

for y in range(5):
    for i in range(20):
        a.add_batch_train_value(5 * i, i, i)
        a.add_batch_test_value(i * 0.8, i, i)
    train, test = a.write_and_req()
    print(train, test)
