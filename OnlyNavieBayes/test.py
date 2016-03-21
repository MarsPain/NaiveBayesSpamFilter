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

# 加载测试数据
filename = '../emails/test/test.txt'
smsWords, classLables = naiveBayes.loadSMSData(filename)
testWordsCount = naiveBayes.setOfWordsToVecTor(vocabularyList, smsWords[0])
testWordsMarkedArray = np.array(testWordsCount)

# 保存！！
pSWi = naiveBayes.bayesTheoremCalcPSWi(pWordsSpamicity, pWordsHealthy, pSpam)

sorted_pSWi_N, sortedWordsMarked_N = \
    naiveBayes.getPreN_pSWi(pSWi, testWordsMarkedArray, N=15)

pSWi = naiveBayes.dealWithRareWords(sorted_pSWi_N, pSpam, sortedWordsMarked_N)

# 计算P(Ci|W)，W为向量。P(Ci|W)只需计算P(W|Ci)P(Ci)
p1 = sum(testWordsMarkedArray * pWordsSpamicity) + np.log(pSpam)
p0 = sum(testWordsMarkedArray * pWordsHealthy) + np.log(1 - pSpam)
if p1 > p0:
    print "spam"
else:
    print 'healthy'
