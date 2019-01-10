from urllib.request import urlopen
from urllib.request import urlretrieve
from bs4 import BeautifulSoup
import re

html=urlopen("https://www.2345.com/?k1029550448")
htmll=BeautifulSoup(html,"lxml")
find_img=htmll.findAll("img",{"class":re.compile("[a-z]*")})
for pri in find_img:
    print(pri.attrs["src"]) 
