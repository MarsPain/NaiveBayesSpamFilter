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

    DS = 0.5 * np.ones(len(vocabularyList))
    DH = 0.5 * np.ones(len(vocabularyList))

    for i in range(iterateNum):
        errorCount = 0.0
        for j in range(testCount):
            testWordsCount = boostNaiveBayes.setOfWordsToVecTor(vocabularyList, testWords[j])
            ps, ph, smsType = boostNaiveBayes.classify(pWordsSpamicity, pWordsHealthy,
                                                       DS, DH, pSpam, testWordsCount)

            if smsType != testWordsType[j]:
                errorCount += 1
                alpha = (ps - ph) / ph
                DS[testWordsCount != 0] = (DS[testWordsCount != 0] * np.exp(alpha))
                DH[testWordsCount != 0] = (DH[testWordsCount != 0] * np.exp(-1 * alpha))

        errorRate = errorCount / testCount
        print '第 %d 轮迭代，错误个数 %d ，错误率 %f' % (i, errorCount, errorRate)
        if errorRate == 0.0:
            break


if __name__ == '__main__':
    training()
