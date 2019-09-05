# Author:  Acer Zhang
# Datetime:2019/9/3
# Copyright belongs to the author.
# Please indicate the source for reprinting


# 首先找出最大、最小项，然后扔到原始列表里0概率符合题意的元素，将剩余的放入一个新列表中，最后使用一层for循环对剩下为数不多的元素进行判断。

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
            min_nums = nums[0]
            max_nums = nums[-1]
            re_list = [i for i in nums if min_nums + i <= target <= max_nums + i]
            for id_i, i in enumerate(re_list):
                tmp = target - i
                if tmp in re_list and (tmp in re_list[re_list.index(tmp) + 1:] or target - 2 * i != 0):
                    return [nums.index(i)+1, nums.index(tmp, nums.index(i) + 1)+1]