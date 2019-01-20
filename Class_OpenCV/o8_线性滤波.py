import cv2 as cv
import Class_OS.o1_获得当前工作目录

#指定路径
path=Class_OS.o1_获得当前工作目录.main()+"Testimg/"
file=path+"RGBimgMin.jpg"

img=cv.imread(file)
cv.imshow("Src",img)

#方框滤波 核-中心权重大
imgBoxFilter=cv.boxFilter(img,-1,(30,30))#第二个参数为输出深度，-1为原图深度 第三个参数为核大小
cv.imshow("BoxFilter",imgBoxFilter)

#均值滤波 核-统一权重
imgBlur=cv.blur(img,(3,3))
cv.imshow("Blur",imgBlur)

#高斯滤波
imgGaussianBlur=cv.GaussianBlur(img,(3,3),0)#第三个参数为在X方向上的偏差
cv.imshow("GaussianBlur",imgGaussianBlur)

cv.waitKey(0)#等待按键后关闭窗口