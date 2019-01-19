import cv2 as cv
import Class_OS.o1_获得当前工作目录

#指定路径
path=Class_OS.o1_获得当前工作目录.main()+"Testimg/"
file=path+"study.jpg"

img=cv.imread(file)
size=img.shape#获取图像形状信息 返回长宽和通道数
print("图像尺寸为："+str(size))

#除法必须得//
imgroi=img[:size[0],:size[1]//2]#高度选取上半段

cv.imshow("1",imgroi)

cv.waitKey(0)#等待按键后关闭窗口
