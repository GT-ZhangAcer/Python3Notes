from urllib.request import urlopen
from urllib.request import urlretrieve
import re

import Class_OS.o1_获得当前工作目录

#指定路径
path=Class_OS.o1_获得当前工作目录.main()+"/imgsave/"

html=urlopen("http://xk.henu.edu.cn")
x=-1
while x<499 :
    find_src="http://xk.henu.edu.cn/cas/genValidateCode?dateTime=Sat%20Jun%2009%202018%2020:16:29%20GMT+0800"
    x+=1
    print("正在抓取第%s个验证码" %x)
    urlretrieve(find_src,path+'%s.jpg' % x)
    
