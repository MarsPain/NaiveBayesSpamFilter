#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_

"""
@Author: MarkLiu
"""
import OnlyNavieBayes.NavieBayes as naiveBayes

# 加载训练好的模型信息
vocabularyList, pWordsSpamicity, pWordsHealthy, pSpam = \
    naiveBayes.getTrainedModelInfo()

# 加载测试数据
filename = '../emails/test/test.txt'
smsWords, classLables = naiveBayes.loadSMSData(filename)

smsType = naiveBayes.classify(vocabularyList, pWordsSpamicity,
                              pWordsHealthy, pSpam, smsWords[0])
print smsType
