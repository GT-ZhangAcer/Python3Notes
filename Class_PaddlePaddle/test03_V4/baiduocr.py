from aip import AipOcr
import Class_OS.o1_获得当前工作目录

#百度AIP接口参数
APP_ID = '15346668'
API_KEY = 'W5ow2FqWF6sGaUX9IAGlrmlw'
SECRET_KEY = 'IdRqWRGOQ7kRndGcdFvY20Y8lyy9r3RQ'

def ocrimage(imgPath):#图像识别调用
    client = AipOcr(APP_ID, API_KEY, SECRET_KEY)
    image=open(imgPath,'rb').read()#非Url
    #info=client.basicGeneralUrl(imgPath)
    info=client.basicGeneral(image)
    info=info['words_result']
    info=info[0]
    info=info['words']
    print("ImageID=="+info)
    return info



#指定路径
path=Class_OS.o1_获得当前工作目录.main()+"dataOri/"

for i in range(500):
    imgpath=path+str(i)+".jpg"
    print("正在识别第" + str(i + 1) + "个图片")
    info=ocrimage(imgpath)
    #解决识别格式错误的数据
    try:
        if (len(info)!= 4):#解决长度出现问题
            info = input(str(i) + "==")
    except:#解决出现特殊符号
        info = input(str(i) + "==")
    with open(path+"ocrData.txt",'at') as f:#wt为不能追加 此处用at
        f.write(info)

print("---OK!")