from aip import AipOcr

APP_ID = ''
API_KEY = ''
SECRET_KEY = ''

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