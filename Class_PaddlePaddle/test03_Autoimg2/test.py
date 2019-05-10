from PIL import Image
import numpy as np
from Class_PaddlePaddle.test03_Autoimg2.torNN import TorNN

import Class_OS.o1_获得当前工作目录


q = [[0, 1, 1], [1, 2, 1], [2, 2, 2], [5, 4, 4], [4, 5, 5], [4, 5, 4], [9, 9, 9]]

a = [[[1, 1, 1], [2, 1, 1], [2, 1, 2], [1, 1, 0]], [[4, 5, 5], [5, 4, 4], [4, 4, 4]]]

obj = TorNN(q, a)
print("metaNorm",obj.metaNorm())
print("obj.p2meta",obj.p2meta())
obj.classsify(expansion_rate=2,debug=True)
