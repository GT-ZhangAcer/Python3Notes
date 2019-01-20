import cv2 as cv
import Class_OS.o1_获得当前工作目录

#指定路径
path=Class_OS.o1_获得当前工作目录.main()+"Testimg/"
file=path+"Dilateimg.jpg"

img=cv.imread(file)
cv.imshow("src",img)

#膨胀
dilateSharp=cv.getStructuringElement(cv.MORPH_RECT,(5,5))#构造膨胀核 第一个参数为核形状 第二个参数为核大小
imgDilate=cv.dilate(img,dilateSharp)
cv.imshow("Dilate",imgDilate)

#腐蚀
imgerode=cv.erode(img,dilateSharp)
cv.imshow("erode",imgerode)

#开运算 闭运算 黑帽 顶帽
imgOpen=cv.morphologyEx(img,cv.MORPH_OPEN,dilateSharp)#开运算 先腐蚀后膨胀
cv.imshow("Open",imgOpen)
imgClose=cv.morphologyEx(img,cv.MORPH_CLOSE,dilateSharp)#闭运算
cv.imshow("Close",imgClose)
imgTopHat=cv.morphologyEx(img,cv.MORPH_TOPHAT,dilateSharp)#顶帽 原图与开运算之差
cv.imshow("TopHat",imgTopHat)
imgBlackHat=cv.morphologyEx(img,cv.MORPH_BLACKHAT,dilateSharp)#黑帽 原图与闭运算之差
cv.imshow("BlackHat",imgBlackHat)

cv.waitKey(0)
