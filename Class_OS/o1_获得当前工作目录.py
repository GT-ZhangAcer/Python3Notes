import sys

def main():
    ScriptFilePath=(sys.argv[0])#获得的是当前执行脚本的位置（若在命令行执行的该命令，则为空）
    WorkPathNum=ScriptFilePath.rindex('/')
    WorkPath=ScriptFilePath[0:WorkPathNum+1]#切片
    print("当前工作目录为："+WorkPath)
    return  WorkPath