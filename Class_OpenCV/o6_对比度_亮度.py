import cv2 as cv
import Class_OS.o1_获得当前工作目录

#指定路径
path=Class_OS.o1_获得当前工作目录.main()+"Testimg/"
file=path+"RGBimgMin.jpg"

img=cv.imread(file)
cv.imshow("Ago",img)
def rgbBringhtAndContrast(cvimg,bringValue,contrastValue):
    ii=0
    for i in cvimg:
        jj=0
        for j in i:
            kk=0
            for k in j:
                cvimg[ii][jj][kk]=contrastValue*cvimg[ii][jj][kk]+bringValue
                kk+=1
            jj+=1
        ii+=1
    print("---OK")

rgbBringhtAndContrast(img,50,0.5)#第二个参数为亮度 第三个参数为对比度，对比度取值为0-100%
cv.imshow("Final",img)

cv.waitKey(0)#等待按键后关闭窗口
'''
目前，网络上大部分使用opencv调整图像对比度和亮度的文章，基本都是源于官网的示例

映射曲线公式为g(x) = a*f(x)+b

公式实际上是没错的，除了上述(f)图外，其他映射曲线都能构造出来。但大部分人却错误地认为a是控制对比度，b是控制亮度的。

对比度：需要通过a 、b 一起控制（仅调整a只能控制像素强度0附近的对比度，而这种做法只会导致像素强度大于0的部分更亮而已，根本看不到对比度提高的效果)

亮度：通过b控制
--------------------- 
作者：abc20002929 
来源：CSDN 
原文：https://blog.csdn.net/abc20002929/article/details/40474807 
版权声明：本文为博主原创文章，转载请附上博文链接！
'''
