import socket
import uuid

# 获取主机名
hostname = socket.gethostname()
#获取IP
ip = socket.gethostbyname(hostname)
# 获取Mac地址
def get_mac_address():
    mac=uuid.UUID(int = uuid.getnode()).hex[-12:]
    return ":".join([mac[e:e+2] for e in range(0,11,2)])

# ipList = socket.gethostbyname_ex(hostname)
# print(ipList)
print("主机名：",hostname)
print("IP：",ip)
print("Mac地址：",get_mac_address())