from PIL import Image
import numpy as np
from Class_PaddlePaddle.test03_Autoimg2.torNN import TorNN

import Class_OS.o1_获得当前工作目录

q = [[1, 1, 1, 7], [0, 0, 0, 6], [3, 3, 3, 5], [4, 4, 4, 4], [5, 5, 5, 4], [6, 6, 6, 2], [7, 7, 7, 1]]
a = [[1, 1, 1, 1], [0, 0, 0, 0], [3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6]]

obj = TorNN(q, a)
print(obj.metaNorm())
print(obj.p2meta())
