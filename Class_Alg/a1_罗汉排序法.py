import threading
import time

def imp(num):
    for i in range(int(num)):
        time.sleep(0.1)
    print(num)

numberlist=[5,6,6,1,8,2,3,10]

for i in numberlist:
    run=threading.Thread(target=imp,args=[i])
    run.start()