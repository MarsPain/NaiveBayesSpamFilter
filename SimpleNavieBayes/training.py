#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_

"""
@Author: MarkLiu
"""
import numpy as np
import SimpleNavieBayes.NavieBayes as naiveBayes

filename = '../emails/training/SMSCollection.txt'
smsWords, classLables = naiveBayes.loadSMSData(filename)
vocabularyList = naiveBayes.createVocabularyList(smsWords)
print "生成语料库！"
trainMarkedWords = naiveBayes.setOfWordsListToVecTor(vocabularyList, smsWords)
print "数据标记完成！"
# 转成array向量
trainMarkedWords = np.array(trainMarkedWords)
print "数据转成矩阵！"
pWordsSpamicity, pWordsHealthy, pSpam = naiveBayes.trainingNaiveBayes(trainMarkedWords, classLables)
print 'pSpam:', pSpam
fpSpam = open('pSpam.txt', 'w')
spam = pSpam.__str__()
fpSpam.write(spam)
fpSpam.close()
# 保存训练生成的语料库信息
# 保存语料库词汇
fw = open('vocabularyList.txt', 'w')
for i in range(len(vocabularyList)):
    fw.write(vocabularyList[i] + '\t')
fw.flush()
fw.close()
# 保存训练阶段获取的参数：pWordsSpamicity和pWordsHealthy
np.savetxt('pWordsSpamicity.txt', pWordsSpamicity, delimiter='\t')
np.savetxt('pWordsHealthy.txt', pWordsHealthy, delimiter='\t')
