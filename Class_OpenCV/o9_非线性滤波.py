import cv2 as cv
import Class_OS.o1_获得当前工作目录

#指定路径
path=Class_OS.o1_获得当前工作目录.main()+"Testimg/"
file=path+"Filterimg.jpg"

img=cv.imread(file)
cv.imshow("Src",img)

#中值滤波 消除噪声有优势，速度比均值滤波慢很多倍
#像素点邻域灰度中值代替该点的灰度值
imgMedianBlur=cv.medianBlur(img,7)#第二个不是核表示，而是线性尺寸，与颜色深度有关
cv.imshow("MedianBlur",imgMedianBlur)

#双边滤波 比高斯滤波多一个高斯方差
#低频处理较好 高频难处理 能较好保留边缘
imgBilateralFilter=cv.bilateralFilter(img,50,25*2,25/2)
#第二个参数为像素邻域直径 第三个为颜色滤波器的范围，越高越容易融合颜色 第四个为坐标滤波器，数值越大变换范围越大
cv.imshow("bilateralFilter",imgBilateralFilter)

cv.waitKey(0)#等待按键后关闭窗口