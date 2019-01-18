import cv2

import Class_OS.获得当前工作目录
#指定路径
path=Class_OS.获得当前工作目录.main()+"Testimg/"
file=path+"study.jpg"

img=cv2.imread(file)#绑定对象
cv2.imshow("Test",img)#展示图片 第一个参数为标题 第二个是被绑定的对象
cv2.waitKey(0)#等待按键后关闭窗口
