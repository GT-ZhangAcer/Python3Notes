import cv2 as cv
import Class_OS.o1_获得当前工作目录

#指定路径
path=Class_OS.o1_获得当前工作目录.main()+"Testimg/"
file=path+"RGBimgMin.jpg"

img=cv.imread(file)
cv.imshow("Ago",img)

#先不写，等理解了之后再说
cv.waitKey(0)#等待按键后关闭窗口