import numpy
import math


class TorNN:
    '''
    TorNN类
    初始化输入类型：
    oridata 1xN1xM 1x图片个数x神经元返回的维度
    M=3 三类:[[0, 1, 1], [1, 2, 1], [2, 2, 1], [5, 4, 4], [4, 5, 5], [4, 5, 4], [9, 9, 9]]
    predata 1xCxNxM 1x类别x图片个数x神经元返回的维度
    M=3 两类:[[[1, 1, 1], [2, 1, 1], [2, 1, 2], [1, 1, 2]], [[4, 5, 5], [5, 4, 4], [4, 4, 4]]]

    classify_scale 分类精度比例 越低越精确
    expansion_multiple
    '''

    def __init__(self, oridata, predata, classify_scale=0.3,expansion_multiple=5):
        self.oriData = oridata
        self.preData = predata
        self.classsify_scale = classify_scale
        self.expansion_multiple=expansion_multiple
        # Dim数据计算
        self.oriDim = len(oridata[0])
        self.preDim = len(predata[0][0])
        if self.oriDim != self.preDim:
            print("神经元返回数据尺寸不符")
            return
        # 图片数量计算
        self.oriNum = len(oridata)
        # 类别数量计算
        # self.classsifyName=[i[1] for i in predata]
        # self.classsifyName=set(self.classsifyName)
        self.classsifyNum = len(predata)

    def metaNorm(self, predata=None):
        if predata is None:
            predata = self.preData
        '''
        各元-欧式距离计算-各取出中心点
        :return: 类别-最小点数据 [[C,M],[...]...]
        '''
        metaNormlist = []
        for data in predata:
            allNormData = []  # 存放各点与各点之间的欧氏距离
            allSumNorm = []  # 存放各点的欧式距离和
            for id, i in enumerate(data):
                '''
                i、ii为每个点的数据
                '''
                i = numpy.array(i)
                dist = []
                for id2, ii in enumerate(data):
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
            metaNormlist.append(data[minID])
            '''
            metaNormlist返回类型 [[fcData],[]...]...
            <class 'list'>: [[1, 1, 1], [4, 4, 4]]
            '''
        return metaNormlist

    def p2meta(self):
        '''
        各元与各点之间的欧氏距离计算
        :return: 计算数据
        '''
        metaNormlist = self.metaNorm()  # 引入元中心点数据
        allNormData = []  # 存放各点与各元之间的欧氏距离

        for id1, i in enumerate(metaNormlist):
            '''
            i为每个元位置数据、ii为每个点的位置数据
            '''
            i = numpy.array(i)
            for id2, ii in enumerate(self.oriData):
                ii = numpy.array(ii)
                normLong = numpy.linalg.norm(i - ii).tolist()
                allNormData.append([id1, id2, float(str(normLong)[:8])])  # 精度为8位
            '''
               allNormData结果类型 [[id1,id2,欧氏距离],[...]...]
               <class 'list'>: [[0, 0, 1.0], [0, 1, 1.0], [0, 2, 1.414213], [0, 3, 5.830951], [0, 4, 6.403124], [0, 5, 5.830951], [0, 6, 13.8564], [1, 0, 5.830951], [1, 1, 4.690415], [1, 2, 4.123105], [1, 3, 1.0], [1, 4, 1.414213], [1, 5, 1.0], [1, 6, 8.660254]]
            '''
        return allNormData

    def classsify(self):
        allNormData = self.p2meta()  # 引入各元与各点之间的欧氏距离
        sortData = sorted(allNormData, key=lambda norm: norm[2])
        print(sortData)
        normNum = [i[2] for i in allNormData]
        print("average", sum(normNum) / len(allNormData))
        print("max", max(normNum))

    def lossPre(self):
        pass
