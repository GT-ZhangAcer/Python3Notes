# Author:  Acer Zhang
# Datetime:2019/9/7
# Copyright belongs to the author.
# Please indicate the source for reprinting.

class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        true_str = ""
        now_char = ""
        if len(strs) == 0:
            return ""
        if len(strs) == 1:
            return strs[0]
        strs.sort()
        first_len = len(strs[0])
        for char_id in range(first_len):
            for item_id, item_str in enumerate(strs):
                if item_id == 0:
                    now_char = item_str[char_id]
                    true_str += now_char
                    continue
                if len(item_str) < char_id + 1:
                    return true_str[:-1]
                if item_str[char_id] == now_char:
                    continue
                else:
                    return true_str[:-1]
        return true_str

