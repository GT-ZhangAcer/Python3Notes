import os
tasklistInfo=os.popen('tasklist /FI "IMAGENAME eq qq.exe"').read()
tasklistBool="QQ" in os.popen('tasklist /FI "IMAGENAME eq qq.exe"').read()
print(tasklistInfo)
print(tasklistBool)