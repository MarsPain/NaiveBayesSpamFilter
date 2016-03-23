#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_

"""
@Author: MarkLiu
"""
import AdaBoostAndNavieBayes.AdaboostNavieBayes as boostNaiveBayes
import random
import numpy as np


def training():
    """
    测试分类的错误率
    :return:
    """
    filename = '../emails/training/SMSCollection.txt'
    smsWords, classLables = boostNaiveBayes.loadSMSData(filename)

    # 交叉验证
    testWords = []
    testWordsType = []

    testCount = 1000
    for i in range(testCount):
        randomIndex = int(random.uniform(0, len(smsWords)))
        testWordsType.append(classLables[randomIndex])
        testWords.append(smsWords[randomIndex])
        del (smsWords[randomIndex])
        del (classLables[randomIndex])

    vocabularyList = boostNaiveBayes.createVocabularyList(smsWords)
    print "生成语料库！"
    trainMarkedWords = boostNaiveBayes.setOfWordsListToVecTor(vocabularyList, smsWords)
    print "数据标记完成！"
    # 转成array向量
    trainMarkedWords = np.array(trainMarkedWords)
    print "数据转成矩阵！"
    pWordsSpamicity, pWordsHealthy, pSpam = \
        boostNaiveBayes.trainingNaiveBayes(trainMarkedWords, classLables)

    iterateNum = 40

    DS = np.ones(len(vocabularyList))

    for i in range(iterateNum):
        errorCount = 0.0
        for j in range(testCount):
            testWordsCount = boostNaiveBayes.setOfWordsToVecTor(vocabularyList, testWords[j])
            ps, ph, smsType = boostNaiveBayes.classify(pWordsSpamicity, pWordsHealthy,
                                                       DS, pSpam, testWordsCount)

            if smsType != testWordsType[j]:
                errorCount += 1
                # alpha = (ph - ps) / ps
                alpha = ps - ph
                if testWordsType[j] == 1:   # 原先为spam，预测成ham
                    DS[testWordsCount != 0] = np.abs((DS[testWordsCount != 0] - np.exp(alpha)) / DS[testWordsCount != 0])
                else:   # 原先为ham，预测成spam
                    DS[testWordsCount != 0] = (DS[testWordsCount != 0] + np.exp(alpha)) / DS[testWordsCount != 0]
        print 'DS:', DS
        errorRate = errorCount / testCount
        print '第 %d 轮迭代，错误个数 %d ，错误率 %f' % (i, errorCount, errorRate)
        if errorRate == 0.0:
            break


if __name__ == '__main__':
    training()
