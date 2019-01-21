import cv2 as cv
import Class_OS.o1_获得当前工作目录

#指定路径
path=Class_OS.o1_获得当前工作目录.main()+"Testimg/"
file=path+"RGBimgmin.jpg"

img=cv.imread(file,0)#二值图读取
cv.imshow("Src",img)

contoursValue=cv.findContours(img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
#第二个参数为检测模式 EXTERNAL只提取最外层 List为提取所有轮廓 CCOMP为两层轮廓 TREE建立轮廓层次
#第三个参数为检测轮廓近似方法 此处默认

finalimg=cv.drawContours(img,contoursValue[0],-200,0)
#第一个参数为原图像 第二个为轮廓线 第三为绘制范围，负数为全部绘制 第四个为颜色 第五个为线条宽度
cv.imshow("Final",finalimg)
cv.waitKey(0)#等待按键后关闭窗口