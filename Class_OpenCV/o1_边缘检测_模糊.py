import cv2
import Class_OS.a1_获得当前工作目录

#指定路径
path=Class_OS.获得当前工作目录.main()+"Testimg/"
file=path+"study.jpg"

img=cv2.imread(file)#绑定对象
cv2.imshow("Test",img)#展示图片 第一个参数为标题 第二个是被绑定的对象
grayimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#转换为灰度图像
blurimg=cv2.blur(grayimg,(3,3))#使用3x3内核降噪

#blurimg2=cv2.blur(img,(7,7))#模糊操作
#cv2.imshow("模糊操作",blurimg2)

cannyimg=cv2.Canny(blurimg,50,150)#第二个参数为最小阈值，第三个为最大阈值
cv2.imshow("边缘检测",cannyimg)
cv2.imwrite(path+"Final.jpg",cannyimg)#写入文件不可以是中文，中文不输出

print(img)
#检测结果反转
ii=0
for i in cannyimg:
    jj=0
    for j in i:
        try:
            if(j==0):
                cannyimg[ii][jj]=255
            else:
                cannyimg[ii][jj]=0
            jj+=1
        except:
            continue
    ii+=1
print("OK")
cv2.imshow("2-边缘检测",cannyimg)
cv2.waitKey(0)#等待按键后关闭窗口
