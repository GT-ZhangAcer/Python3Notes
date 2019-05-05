from PIL import Image
import numpy as np
from Class_PaddlePaddle.test03_Autoimg2.torNN import TorNN

import Class_OS.o1_获得当前工作目录

q = [[1, 1, 1, 1], [0, 0, 0, 0], [3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7]]
a = [[1, 1, 1, 1], [0, 0, 0, 0], [3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7]]

obj = TorNN(q, a)
print(obj.metaNorm())
