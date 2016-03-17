#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_

"""
@Author: MarkLiu
"""
import numpy as np
import OnlyNavieBayes.NaiveBayes as util

filename = '../emails/training/SMSCollection.txt'
smsWords, classLables = util.loadSMSData(filename)
vocabularyList = util.createVocabularyList(smsWords)
print "生成语料库！"
trainMarkedWords = []
for words in smsWords:
    vocabMarked = util.setOfWordsToVecTor(vocabularyList, words)
    trainMarkedWords.append(vocabMarked)
print "数据标记完成！"
# 转成array向量
trainMarkedWords = np.array(trainMarkedWords)
print "数据转成矩阵！"
pWordsSpamicity, pWordsHealthy, pSpam = util.trainingNaiveBayes(trainMarkedWords, classLables)
print 'pSpam:', pSpam

# 保存训练生成的语料库信息
# 保存语料库词汇
fw = open('vocabularyList.txt', 'w')
for i in range(len(vocabularyList)):
    fw.write(vocabularyList[i] + '\t')

# 保存pWordsSpamicity和pWordsHealthy
np.savetxt('pWordsSpamicity.txt', pWordsSpamicity, delimiter='\t')
np.savetxt('pWordsHealthy.txt', pWordsHealthy, delimiter='\t')
