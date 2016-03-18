#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_

"""
@Author: MarkLiu
"""

import numpy as np
import OnlyNavieBayes.NavieBayes as naiveBayes

# 加载训练获取的语料库信息
vocabularyList = naiveBayes.getVocabularyList('vocabularyList.txt')
pWordsHealthy = np.loadtxt('pWordsHealthy.txt', delimiter='\t')
pWordsSpamicity = np.loadtxt('pWordsSpamicity.txt', delimiter='\t')
fr = open('pSpam.txt')
pSpam = float(fr.readline().strip())
print 'pSpam:', type(pSpam), pSpam

# 保存！！
pSWi, pHWi = naiveBayes.bayesTheoremCalc(pWordsSpamicity, pWordsHealthy, pSpam)

# 加载测试数据
filename = '../emails/test/test.txt'
smsWords, classLables = naiveBayes.loadSMSData(filename)
testWordsCount = naiveBayes.setOfWordsToVecTor(vocabularyList, smsWords[0])
testWordsCountArray = np.array(testWordsCount)
result = naiveBayes.classify(pSWi, pHWi, pSpam, testWordsCountArray)
print 'result:', result
