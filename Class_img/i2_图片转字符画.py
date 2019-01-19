#coding:utf-8
from PIL import Image

path="C:/PythonTemp/"#系统绝对路径
filename="1.png"

def convert(img):
    #要索引的字符列表
    ascii_char = list("$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. ")
    length = len(ascii_char)
    img = img.convert("L") # 转为灰度图像
    txt = "" 
    for i in range(img.size[1]): 
        for j in range(img.size[0]):
            gray = img.getpixel((j, i)) # 获取每个坐标像素点的灰度
            unit = 256.0 / length
            txt += ascii_char[int(gray / unit)]+" " #获取对应坐标的字符值
        txt += '\n' 
    return txt 

def MainScript(path,filename):
    img = Image.open(path+filename)#读取图像文件
    (width,height) = img.size
    img = img.resize((int(width*0.9),int(height*0.5))) #对图像进行一定缩小 
    txt = convert(img)
    f = open(path+filename+"_convert.txt","w") 
    f.write(txt) #存储到文件中 
    f.close()
    print("制作完毕,文本已经保存在"+path+filename)

MainScript(path,filename)