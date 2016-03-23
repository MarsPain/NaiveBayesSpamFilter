#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_

"""
@Author: MarkLiu
"""
import numpy as np

import AdaBoostAndNavieBayes.AdaboostNavieBayes as boostNaiveBayes


def getTrainAdaboostInfo():
    """
    获取训练算法阶段的DS和minErrorRate信息
    :return:
    """
    trainDS = np.loadtxt('trainDS.txt', delimiter='\t')
    trainMinErrorRate = np.loadtxt('trainMinErrorRate.txt', delimiter='\t')
    vocabularyList = boostNaiveBayes.getVocabularyList('vocabularyList.txt')
    pWordsSpamicity = np.loadtxt('pWordsSpamicity.txt', delimiter='\t')
    pWordsHealthy = np.loadtxt('pWordsHealthy.txt', delimiter='\t')
    pSpam = np.loadtxt('pSpam.txt', delimiter='\t')
    return vocabularyList, pWordsSpamicity, pWordsHealthy, pSpam, trainMinErrorRate, trainDS


def simpleTest():
    # 加载训练好的模型信息
    vocabularyList, pWordsSpamicity, pWordsHealthy, pSpam, trainMinErrorRate, trainDS = \
        getTrainAdaboostInfo()

    # 加载测试数据
    filename = '../emails/test/test.txt'
    smsWords, classLables = boostNaiveBayes.loadSMSData(filename)
    testWordsMarkedArray = \
        boostNaiveBayes.setOfWordsToVecTor(vocabularyList, smsWords[0])
    ps, ph, smsType = boostNaiveBayes.classify(
            pWordsSpamicity, pWordsHealthy, trainDS, pSpam, testWordsMarkedArray)
    print smsType


if __name__ == '__main__':
    simpleTest()
