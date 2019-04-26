from PIL import Image

class imgSize:
    def __init__(self,im):
        self.im=im
    #等比例拉伸图片
    def imgBig(self,xNum):
        im=Image.open(self.im)
        w,h=im.size
        im=im.resize(w*xNum,h*xNum)
        return im
    #等比例缩小图片
    def imgSmall(self,xNum):
        im=Image.open(self.im)
        w,h=im.size
        im=im.resize(w//xNum,h//xNum)
        return im
