from PIL import Image
import Class_OS.o1_获得当前工作目录

# 指定路径
path = Class_OS.o1_获得当前工作目录.main()
pathFinal = Class_OS.o1_获得当前工作目录.main() + "data/"
x = 15  # 偏移量
print("---Start")

for i in range(500):
    file = path + "dataOri/" + str(i) + ".jpg"
    img = Image.open(file)
    img1 = img.crop((6, 0, 6 + x, 30))
    img2 = img.crop((6 + x, 0, 6 + 2 * x, 30))
    img3 = img.crop((6 + 2 * x, 0, 6 + 3 * x, 30))
    img4 = img.crop((6 + 3 * x, 0, 6 + 4 * x, 30))
    img1.save(pathFinal + str(4 * (i) + 1) + ".jpg")
    img2.save(pathFinal + str(4 * (i) + 2) + ".jpg")
    img3.save(pathFinal + str(4 * (i) + 3) + ".jpg")
    img4.save(pathFinal + str(4 * (i) + 4) + ".jpg")
print("img---OK")


