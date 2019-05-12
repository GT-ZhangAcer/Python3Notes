import numpy
import math


class TorNN:
    '''
    TorNN类
    初始化输入类型：
    oridata 1xN1xM 1x图片个数x神经元返回的维度
    M=3 三类:[[0, 1, 1], [1, 2, 1], [2, 2, 1], [5, 4, 4], [4, 5, 5], [4, 5, 4], [9, 9, 9]]
    predata 1xCxNxM 1x类别Labelx图片个数x神经元返回的维度
    M=3 两类:[[[1, 1, 1], [2, 1, 1], [2, 1, 2], [1, 1, 2]], [[4, 5, 5], [5, 4, 4], [4, 4, 4]]]

    expansion_rate 扩充效率 默认为1
    '''

    def __init__(self, oridata, predata, expansion_rate=1):
        self.oriData = oridata
        self.preData = predata
        self.expansion_rate = expansion_rate
        # Dim数据计算
        self.oriDim = len(oridata[0])
        self.preDim = len(predata[0][0])
        assert self.oriDim == self.preDim, "神经元返回数据尺寸不相同"

        # 图片数量计算
        self.oriNum = len(oridata)
        print("|总需要分类的数据个数为:", self.oriNum)
        # 类别数量计算
        self.classsifyNum = len(predata)

    def metaNorm(self, predata=None):
        if predata is None:
            predata = self.preData
        '''
        各元-欧式距离计算-各取出中心点、最远点
        :return: 类别-最小点数据 [[C,M],[...]...]
        '''
        metaNormMin = []
        metaNormMax = []
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
            metaNormMin.append(data[minID])
            minNorm = str(max(allSumNorm))[:-1]
            minID = [str(i)[:-1] for i in allSumNorm]  # 转换str类型 防止数字变动
            minID = minID.index(minNorm)  # 找出最中心点
            metaNormMax.append(data[minID])
            '''
            metaNormlist返回类型 [[fcData1Min],[fcData2Min],[fcData1Max],[fcData2Max]...]...
            <class 'list'>: [[1, 1, 1], [4, 4, 4]], [[2, 1, 2], [4, 5, 5]]
            '''
        return metaNormMin, metaNormMax

    def p2meta(self):
        '''
        各元与各点之间的欧氏距离计算|各点与点之间欧氏距离
        :return: 计算数据
        '''
        metaNormlist = self.metaNorm()[0]  # 引入元最近中心点数据
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
               allNormData结果类型 [[元id,点id,欧氏距离],[...]...]
               <class 'list'>: [[0, 0, 1.0], [0, 1, 1.0], [0, 2, 1.414213], [0, 3, 5.830951], [0, 4, 6.403124], [0, 5, 5.830951], [0, 6, 13.8564], [1, 0, 5.830951], [1, 1, 4.690415], [1, 2, 4.123105], [1, 3, 1.0], [1, 4, 1.414213], [1, 5, 1.0], [1, 6, 8.660254]]
            '''
        return allNormData

    def p2p(self, aList, bList):
        '''
        点与对应点之间欧氏距离
        :param aList: 点a列表 [[1, 1, 1], [4, 4, 4]]
        :param bList: 点b列表 [[2, 1, 2], [4, 5, 5]]
        :return: 欧氏距离列表[1,2,3]
        '''
        assert len(aList) == len(bList), "点数据非一一对应关系"
        normLong = []
        for i in range(len(aList)):
            a = numpy.array(aList[i])
            b = numpy.array(bList[i])
            normLong.append(numpy.linalg.norm(a - b).tolist())
        return normLong

    def classsify(self, expansion_rate=None,debug=None):
        '''

        :param expansion_rate: 扩张率 默认为1 越大学习越快，准确率越低
        :return:[[已分类:[标签，点ID],[标签，点ID],...],[未分类:[点ID],...]]
        <class 'list'>: [[0, 0], [0, 1], [1, 3], [1, 5], [1, 4], [0, 2]],[[6]]
        '''
        classifyTrue = []
        classifyFalse = []
        if expansion_rate is None:
            expansion_rate = self.expansion_rate

        metaNormMin, metaNormMax = self.metaNorm()  # 引入最远与中心点的点数据
        metaNormList = [i * expansion_rate for i in self.p2p(metaNormMax, metaNormMin)]  # 引入最大可承受距离列表
        allNormData = self.p2meta()  # 引入各元与各点之间的欧氏距离
        sortData = sorted(allNormData, key=lambda norm: norm[2])  # 按距离值从小到大排序


        tureNum = 0  # 成功分类的数量
        pointIDTrue=[]
        for i in sortData:
            metaID, pointID, thisNorm = i
            if (metaNormList[metaID] >= thisNorm) and (pointID not in pointIDTrue):
                classifyTrue.append([metaID, pointID])
                pointIDTrue.append(pointID)
                tureNum += 1
            elif pointID not in pointIDTrue:
                classifyFalse.append(pointID)
        classifyFalse=list(set(classifyFalse))
        falseNum = self.oriNum - tureNum
        if debug is not None:
            print("|TorNN|成功分类个数：", tureNum, "未分类数：", falseNum)
            print("|成功：",classifyTrue)
            print("|未分类：",classifyFalse)
            sortData2 = sorted(allNormData, key=lambda norm: norm[1])
        return classifyTrue,classifyFalse
    def lossPre(self):
        pass
