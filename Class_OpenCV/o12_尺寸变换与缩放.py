import cv2 as cv
import Class_OS.o1_获得当前工作目录

#指定路径
path=Class_OS.o1_获得当前工作目录.main()+"Testimg/"
file=path+"RGBimgMin.jpg"

img=cv.imread(file)
cv.imshow("Src",img)

img2=cv.pyrUp(img)#向上采样
cv.imshow("UP",img2)
img3=cv.pyrDown(img)#向上采样
cv.imshow("Down",img3)

img4=cv.resize(img,(200,200),interpolation=cv.INTER_AREA)#区域差值
img5=cv.resize(img,(200,200))#默认 像素点明显
cv.imshow("Img4-1",img4)
cv.imshow("img4-src",img5)
cv.waitKey(0)