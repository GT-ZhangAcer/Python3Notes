import numpy
import math


class TorNN:
    '''
    TorNN类
    初始化输入类型：
    oridata 1xN1x1+M 1x图片个数xID+神经元返回的维度
    predata 1xN1x1+1+M 1x图片个数xID+标签+神经元返回的维度
    '''

    def __init__(self, oridata, predata):
        self.oriData = oridata
        self.preData = predata
        # Dim数据计算
        self.oriDim = len(oridata[0]) - 1
        self.preDim = len(predata[0]) - 2
        # 图片数量计算
        self.oriNum = len(oridata)
        self.preNum = len(predata)

    def metaNorm(self):
        '''
        元-欧式距离计算并取出最小点
        :return: 最小点数据 1xM
        '''
        allNormData = []  # 存放各点与各点之间的欧氏距离
        allSumNorm = []  # 存放各点的欧式距离和
        for id, i in enumerate(self.preData):
            i = numpy.array(i[2:])
            dist = []
            for id2, ii in enumerate(self.preData):
                if id == id2:
                    continue
                ii = numpy.array(ii[2:])
                # 求欧氏距离
                normLong = numpy.linalg.norm(i - ii).tolist()
                dist.append(float(str(normLong)[:8]))  # 精度为8位
            allNormData.append(dist)
            allSumNorm.append(math.fsum(dist))
        minNorm = str(min(allSumNorm))[:-1]
        minID = [str(i)[:-1] for i in allSumNorm]
        minID = minID.index(minNorm)  # 找出最中心点
        return self.preData[minID]
