from urllib.request import urlopen
from urllib.request import urlretrieve
from bs4 import BeautifulSoup
import re

html=urlopen("https://www.2345.com/?k1029550448")
htmll=BeautifulSoup(html,"lxml")
find_img=htmll.findAll("img",{"class":re.compile("[a-z]*")})
x=0
for pri in find_img:
    x=x+1
    print(pri.attrs["src"])
    find_src="https://www.2345.com/" + pri.attrs["src"]
    print(find_src)
    urlretrieve(find_src,str(x) + ".png")
    
