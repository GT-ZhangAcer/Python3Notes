import cv2 as cv
import Class_OS.o1_获得当前工作目录

#指定路径
path=Class_OS.o1_获得当前工作目录.main()+"Testimg/"
file=path+"cloth.jpg"

img=cv.imread(file)
cv.imshow("Src",img)

cannyimg=cv.Canny(img,50,150)#第二个参数为最小阈值，第三个为最大阈值
cv.imshow("Cannyimg",cannyimg)
sobleimg=cv.Sobel(img,-1,1,0)#第二个参数为图像深度 第三个和第四个为XY方向上的差分阶数
cv.imshow("Sobel",sobleimg)
laplacianimg=cv.Laplacian(img,-1)#拉普拉斯算子
cv.imshow("laplacian",laplacianimg)
scharrimg=cv.Scharr(img,-1,1,0)#第二个参数为图像深度 第三个和第四个为XY方向上的差分阶数
cv.imshow("scharr",sobleimg)
cv.waitKey(0)