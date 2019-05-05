import numpy
import math


class TorNN:
    '''
    TorNN类
    初始化输入类型：
    oridata 1xN1xM 1x图片个数xID+神经元返回的维度
    predata 1xN1xM 1x图片个数xID+神经元返回的维度
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

    def metaNorm(self, predata=None):
        if predata is None:
            predata = self.preData
        '''
        元-欧式距离计算-取出最小点
        :return: 最小点数据 1xM
        '''
        allNormData = []  # 存放各点与各点之间的欧氏距离
        allSumNorm = []  # 存放各点的欧式距离和
        for id, i in enumerate(predata):
            '''
            i、ii为每个点的数据
            '''
            i = numpy.array(i)
            dist = []
            for id2, ii in enumerate(predata):
                if id == id2:
                    continue
                ii = numpy.array(ii)
                # 求欧氏距离
                normLong = numpy.linalg.norm(i - ii).tolist()
                dist.append(float(str(normLong)[:8]))  # 精度为8位
            allNormData.append(dist)
            allSumNorm.append(math.fsum(dist))
        minNorm = str(min(allSumNorm))[:-1]
        minID = [str(i)[:-1] for i in allSumNorm]  # 转换str类型 防止数字变动
        minID = minID.index(minNorm)  # 找出最中心点
        return self.preData[minID]

    def p2meta(self):
        '''
        各元与各点之间的欧氏距离计算
        :return: 计算数据
        '''
        allNormData = []  # 存放各点与各元之间的欧氏距离

        for id1, i in enumerate(self.preData):
            '''
            i为每个元位置数据、ii为每个点的位置数据
            '''
            i = numpy.array(i)
            dist = []
            '''
            dist结果类型 [[id,欧氏距离]...]
            <class 'list'>: [[0, 1.73205], [1, 1.0], [2, 5.567764], [3, 7.549834], [4, 9.539392], [5, 11.53256], [6, 13.52774]]
            '''
            for id2, ii in enumerate(self.oriData):
                ii = numpy.array(ii)
                normLong = numpy.linalg.norm(i - ii).tolist()
                dist.append([id2, float(str(normLong)[:8])])  # 精度为8位
            allNormData.append([id1, dist])
            '''
               allNormData结果类型 [[id1,[id2,欧氏距离],[id2,欧氏距离]...]...]
               <class 'list'>: [[0, [[0, 0.0], [1, 2.0], [2, 4.0], [3, 6.0], [4, 8.0], [5, 10.0], [6, 12.0]]], [1, [[0, 2.0], [1, 0.0], [2, 6.0], [3, 8.0], [4, 10.0], [5, 12.0], [6, 14.0]]], [2, [[0, 4.0], [1, 6.0], [2, 0.0], [3, 2.0], [4, 4.0], [5, 6.0], [6, 8.0]]], [3, [[0, 6.0], [1, 8.0], [2, 2.0], [3, 0.0], [4, 2.0], [5, 4.0], [6, 6.0]]], [4, [[0, 8.0], [1, 10.0], [2, 4.0], [3, 2.0], [4, 0.0], [5, 2.0], [6, 4.0]]], [5, [[0, 10.0], [1, 12.0], [2, 6.0], [3, 4.0], [4, 2.0], [5, 0.0], [6, 2.0]]]]
            '''
        return allNormData

    def lossPre(self):
        self.metaNorm()
