import cv2
import Class_OS.o1_获得当前工作目录

#指定路径
path=Class_OS.o1_获得当前工作目录.main()+"dataOri/"
pathFinal=Class_OS.o1_获得当前工作目录.main()+"datawb/"
for i in range(500):
    file=path+str(i)+".jpg"
    img=cv2.imread(file)
    imggray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#转换为灰度图像
    cv2.imwrite(pathFinal+str(i)+".jpg",imggray)
print("---OK")