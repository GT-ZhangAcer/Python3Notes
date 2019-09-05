# Author:  Acer Zhang
# Datetime:2019/9/4
# Copyright belongs to the author.
# Please indicate the source for reprinting.

def reverse(x):
    str_x = str(abs(x))
    new_str = ""
    for i in range(1, len(str_x) + 1):
        new_str += str_x[-i]
    new_int = int(new_str)
    if new_int < (-2 ** 31) or new_int > (2 ** 31 - 1):
        return 0
    if x != abs(x):
        new_int = -new_int
    return new_int


print(reverse(123))
