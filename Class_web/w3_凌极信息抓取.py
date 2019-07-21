from urllib.request import urlopen
from urllib import request
from bs4 import BeautifulSoup
import json
import time
import csv

key = ['学号', '姓名']

timea = time.strftime("%Y-%m-%d-%H-%M", time.localtime())  # 记录时间
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:56.0) Gecko/20100101 Firefox/56.0'}  # 全局UA


def findname(id):
    url = 'http://202.196.97.26/Platform/AccountInfo?accountName=' + id
    req = request.Request(url, headers=headers)
    html = urlopen(req)
    html_BSObj = BeautifulSoup(html, "lxml")
    find_json = str(html_BSObj.findAll("p")).replace("</p>", "").replace("<p>", "")
    info = json.loads(find_json)
    name = info[0]['displayName']
    return name


sum = 0
falsesum = 0
start = input("start_")
end = input("end_")
with open("./w3/" + str(timea) + ".csv", 'w', newline='', encoding='utf-8')as f:
    writer = csv.DictWriter(f, key)
    writer.writeheader()
    for years in range(int(start[:3]), int(end[:3])):

        try:
            nameL = findname(str(years))
            finalInfo = {'学号': years, '姓名': nameL}
            writer.writerow(finalInfo)

        except:
            falsesum += 1
            print(years, "ERROR")
        sum += 1

print(sum, falsesum)
