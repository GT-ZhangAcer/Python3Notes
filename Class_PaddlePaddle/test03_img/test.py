from PIL import Image


import Class_OS.o1_获得当前工作目录


# 指定路径
path = Class_OS.o1_获得当前工作目录.main()
shape = [3, 30, 15]
im = Image.open(path + "data/5.jpg")
for i in range(5):
    im = im.resize((shape[2] // 3, shape[1] // 3), Image.ANTIALIAS)
    im = im.resize((shape[2], shape[1]), Image.ANTIALIAS)
im.show()