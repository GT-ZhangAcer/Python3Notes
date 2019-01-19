import os

netinfo=os.popen('net view').read()
print(netinfo)
