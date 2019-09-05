# Author:  Acer Zhang
# Datetime:2019/9/4
# Copyright belongs to the author.
# Please indicate the source for reprinting.

class Solution:
    def isPalindrome(self, x: int) -> bool:
        list_x = [i for i in str(x)]
        list_y = list(list_x)
        list_x.reverse()
        if list_x == list_y:
            return True
        else:
            return False

a = 123
print(isPalindrome(20302))
