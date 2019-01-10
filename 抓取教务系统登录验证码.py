from urllib.request import urlopen
from urllib.request import urlretrieve
import re

html=urlopen("http://xk.henu.edu.cn")
x=-1
while x<=500 :
    find_src="http://xk.henu.edu.cn/cas/genValidateCode?dateTime=Sat%20Jun%2009%202018%2020:16:29%20GMT+0800"
    x+=1
    print("正在抓取第%s个验证码" %x)
    urlretrieve(find_src,'img/%s.jpg' % x)
    
