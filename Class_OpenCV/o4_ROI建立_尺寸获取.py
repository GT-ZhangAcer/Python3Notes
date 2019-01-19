import cv2 as cv
import Class_OS.o1_获得当前工作目录

#指定路径
path=Class_OS.o1_获得当前工作目录.main()+"Testimg/"
file=path+"study.jpg"

img=cv.imread(file)
size=img.shape#获取图像形状信息 返回长宽和通道数
print("图像尺寸为："+str(size))

#切割图像
#除法必须得a//b
imgroi=img[:size[0],:size[1]//2]#宽度选取上半段
imgroi2=img[:size[0],:size[1]//2]
cv.imshow("1",imgroi)

#拼合图像
addimg=cv.addWeighted(imgroi,0.3,imgroi2,0.1,0)#第一个参数为底图 第二个参数为底图透明度 第三个和第四个同理为贴图 第五个为Gamma，权重计算后的附加值
cv.imshow("2",addimg)
cv.waitKey(0)#等待按键后关闭窗口
