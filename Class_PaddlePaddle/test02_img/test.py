from PIL import Image
import Class_OS.o1_获得当前工作目录
import numpy as np

path=Class_OS.o1_获得当前工作目录.main()

'''lst = list()
for i in range(5):
    image = Image.open(path++"datawb/"str(i)+".jpg")
    lst.append(np.array(image))'''

'''textlst=list()#文字列表
text=open(path+"dataOri/ocrData.txt","r")
for line in text:
    textlst.append(np.array(line[:4]))
print(textlst)'''

