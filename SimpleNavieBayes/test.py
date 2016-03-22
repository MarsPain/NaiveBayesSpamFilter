#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_

"""
@Author: MarkLiu
"""
import SimpleNavieBayes.NavieBayes as naiveBayes
import random
import numpy as np


def simpleTest():
    # 加载训练好的模型信息
    vocabularyList, pWordsSpamicity, pWordsHealthy, pSpam = \
        naiveBayes.getTrainedModelInfo()

    # 加载测试数据
    filename = '../emails/test/test.txt'
    smsWords, classLables = naiveBayes.loadSMSData(filename)

    smsType = naiveBayes.classify(vocabularyList, pWordsSpamicity,
                                  pWordsHealthy, pSpam, smsWords[0])
    print smsType


def testClassifyErrorRate():
    """
    测试分类的错误率
    :return:
    """
    filename = '../emails/training/SMSCollection.txt'
    smsWords, classLables = naiveBayes.loadSMSData(filename)

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

    vocabularyList = naiveBayes.createVocabularyList(smsWords)
    print "生成语料库！"
    trainMarkedWords = naiveBayes.setOfWordsListToVecTor(vocabularyList, smsWords)
    print "数据标记完成！"
    # 转成array向量
    trainMarkedWords = np.array(trainMarkedWords)
    print "数据转成矩阵！"
    pWordsSpamicity, pWordsHealthy, pSpam = naiveBayes.trainingNaiveBayes(trainMarkedWords, classLables)

    errorCount = 0.0
    for i in range(testCount):
        smsType = naiveBayes.classify(vocabularyList, pWordsSpamicity,
                                      pWordsHealthy, pSpam, testWords[i])
        print '预测类别：', smsType, '实际类别：', testWordsType[i]
        if smsType != testWordsType[i]:
            errorCount += 1

    print '错误个数：', errorCount, '错误率：', errorCount / testCount


if __name__ == '__main__':
    testClassifyErrorRate()
