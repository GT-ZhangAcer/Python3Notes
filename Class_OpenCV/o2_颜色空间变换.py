import cv2

import Class_OS.o1_获得当前工作目录

#指定路径
path=Class_OS.o1_获得当前工作目录.main()+"Testimg/"
file=path+"study.jpg"

img=cv2.imread(file)
imggray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#转换为灰度图像
imgRGB=cv2.cvtColor(imggray,cv2.COLOR_GRAY2RGB)#再转换为RGB图像
cv2.imshow("Gray",imggray)
cv2.imshow("RGB",imgRGB)

cv2.waitKey(0)#等待按键后关闭窗口