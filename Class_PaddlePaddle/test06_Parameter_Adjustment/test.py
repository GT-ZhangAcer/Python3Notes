# Author: Acer Zhang
# Datetime:2019/8/25 17:02
# Copyright belongs to the author.
# Please indicate the source for reprinting.
import numpy as np

import script.log_Script as log

a = log.WriteLog()

for i in range(10):
    for ii in range(15):
        ii = np.array([ii])
        a.add_batch_train_value(ii / 100, ii / 200, ii / 100)
    print(a.write_and_req())
