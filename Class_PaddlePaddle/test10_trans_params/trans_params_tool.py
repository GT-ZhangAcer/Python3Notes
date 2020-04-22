# Author: Acer Zhang
# Datetime:2020/3/23 21:55
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import os

ori_params_path = r"D:\a13\server-python\ERNIE\params"
new_name = "R"

file_list = os.listdir(ori_params_path)
print("Load", len(file_list), "files")
for file_name in file_list:
    ori_file_path = os.path.join(ori_params_path, file_name)
    new_file_path = os.path.join(ori_params_path, file_name)
    if file_name == "@LR_DECAY_COUNTER@":
        continue
    os.rename(ori_file_path, new_file_path)
print("Done")
