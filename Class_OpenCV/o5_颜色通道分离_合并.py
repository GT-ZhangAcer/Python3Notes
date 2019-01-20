import cv2 as cv
import Class_OS.o1_获得当前工作目录

#指定路径
path=Class_OS.o1_获得当前工作目录.main()+"Testimg/"
file=path+"RGBimgMin.jpg"

img=cv.imread(file)
imgsplit=cv.split(img)#通道分离
cv.imshow("R",imgsplit[0])
imgmerge=cv.merge(imgsplit)
cv.imshow("RGB",imgmerge)
cv.waitKey(0)#等待按键后关闭窗口