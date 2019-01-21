import cv2 as cv
import Class_OS.o1_获得当前工作目录

#指定路径
path=Class_OS.o1_获得当前工作目录.main()+"Testimg/"
file=path+"Equalimg.jpg"

img=cv.imread(file)
cv.imshow("Src",img)
img2=cv.cvtColor(img,cv.COLOR_BGR2GRAY)#转化为灰度图像
cv.imshow("Gray",img2)
finalimg=cv.equalizeHist(img2)#直方图均衡化 仅支持8bit图像
cv.imshow("Final",finalimg)

#转换为RGB图像方法
imgsplit=cv.split(img)#通道分离
imgsplit[0]=cv.equalizeHist(imgsplit[0])#分别进行直方图均衡化
imgsplit[1]=cv.equalizeHist(imgsplit[1])
imgsplit[2]=cv.equalizeHist(imgsplit[2])
finalimg2=cv.merge(imgsplit)#合并通道
cv.imshow("Final",finalimg2)

cv.waitKey(0)