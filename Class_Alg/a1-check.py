import time

class Check():
    def __init__(self,M=12,D=20):
        self.m

    def check(self,key):
        YCheck = ["A", "S", "L", "K", "G"]
        MCheck = ["Q", "R", "C", "V", "B"]
        DCheck = ["T", "Y", "W", "U", "I"]
        LNUM = {"D": 1, "E": 2, "F": 5, "H": 10, "J": 15, "M": 20, "X": 25, "O": 11, "P": 18}
        try:
            for i in YCheck:
                key = str(key).replace(i, "")
            YKey = key[:2]
            YKey = LNUM[YKey[0]] + LNUM[YKey[1]]
            key = key[2:]

            for i in MCheck:
                key = str(key).replace(i, "")
            MKey = key[:2]
            MKey = LNUM[MKey[0]] + LNUM[MKey[1]]
            key = key[2:]
            for i in DCheck:
                key = str(key).replace(i, "")
            DKey = key[:2]
            DKey = LNUM[DKey[0]] + LNUM[DKey[1]]
            key = key[2:]
        except:
            print("==========\n密钥不正确请重新配置！\n==========")
            while (1):
                time.sleep(5000)

        timea = str(time.strftime("%Y,%m,%d", time.localtime())).split(",")
        """
    
    
        """
        if (int(YKey) == 19 and int(timea[1]) <= MKey and int(timea[2]) <= DKey):
            print("==========\nOK\n==========")
        else:
            print("==========\nERROR\n==========")
            while (1):
                time.sleep(50)

    def make(self,M,D):

