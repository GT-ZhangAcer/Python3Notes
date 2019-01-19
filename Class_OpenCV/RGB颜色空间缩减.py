import cv2

def rgbSpaceMin(cvimg):
    ii=0
    for i in cvimg:
        jj=0
        for j in i:
            kk=0
            for k in j:
                cvimg[ii][jj][kk]=k/10
                kk+=1
            jj+=1
        ii+=1
    print("颜色空间压缩---OK")

def graySpaceMin(cvimg):
    ii=0
    for i in cvimg:
        jj=0
        for j in i:
            cvimg[ii][jj]=j/10
            jj+=1
        ii+=1
    print("颜色空间压缩---OK")