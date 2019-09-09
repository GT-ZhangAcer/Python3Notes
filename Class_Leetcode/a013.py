# Author:  Acer Zhang
# Datetime:2019/9/5
# Copyright belongs to the author.
# Please indicate the source for reprinting.

# s = "LVIII" # 58
s = "MCMXCIV"  # 1994

"""
    整体思路从右往左开始索引，如果出现特殊情况则越过特殊情况的字符，除此之外均使右边与左边进行比较。
"""
roman_map = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
max_str_num = len(s)
num = roman_map[s[max_str_num - 1]]
skip_flag = False
for i in range(1, max_str_num):
    if skip_flag:
        skip_flag = False
        num += roman_map[s[max_str_num - i - 1]]
        print(roman_map[s[max_str_num - i - 1]])
        continue

    lift = roman_map[s[max_str_num - i - 1]]  # 左边字符值
    right = roman_map[s[max_str_num - i]]

    if lift >= right:
        num += lift
    else:
        num -= lift
        skip_flag = True
    print(num, lift, right, skip_flag)
print(num)
