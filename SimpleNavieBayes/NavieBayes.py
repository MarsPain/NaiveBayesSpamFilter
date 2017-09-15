#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_

"""
@Author: MarkLiu
"""
import numpy as np

#对SMS预处理，去除空字符串，并统一小写
def textParser(text):
    import re
    regEx = re.compile(r'[^a-zA-Z]|\d')  # 匹配非字母或者数字，即去掉非字母非数字，只留下单词
    words = regEx.split(text)
    # 去除空字符串，并统一小写
    words = [word.lower() for word in words if len(word) > 0]
    return words

#加载SMS数据
def loadSMSData(fileName):
    print(fileName)
    f = open(fileName)
    # 类别标签，1表示是垃圾SMS，0表示正常SMS
    classCategory = []
    # 文本域
    smsWords = []
    # 切分文本

    for line in f.readlines():
        linedatas = line.strip().split('\t')    #strip删除两边的空白字符，split按照空白字符对文本进行分割
        if linedatas[0] == 'ham':
            classCategory.append(0)
        elif linedatas[0] == 'spam':
            classCategory.append(1)
        words = textParser(linedatas[1])    #此处的linedatas[1]难道不是第一个单词？
        smsWords.append(words)
    return smsWords, classCategory

#创建词库
def createVocabularyList(smsWords):
    vocabularySet = set([])
    for words in smsWords:
        # set()会返回一个不重复的词表，在迭代中用操作符|得到两个集合的并集，从而去除重复的词
        vocabularySet = vocabularySet | set(words)
    vocabularyList = list(vocabularySet)
    return vocabularyList


def getVocabularyList(fileName):
    """
    从词汇列表文件中获取语料库
    :param fileName:
    :return:
    """
    fr = open(fileName).encoding("utf-8")
    vocabularyList = fr.readline().strip().split('\t')
    fr.close()
    return vocabularyList

# 构建词向量（特征提取），用邮件内容匹配词库，标记词库中的词汇出现的次数
def setOfWordsToVecTor(vocabularyList, smsWords):
    vocabMarked = [0] * len(vocabularyList)
    for smsWord in smsWords:
        if smsWord in vocabularyList:
            vocabMarked[vocabularyList.index(smsWord)] += 1
    return vocabMarked

# 将词向量存储进矩阵
def setOfWordsListToVecTor(vocabularyList, smsWordsList):
    vocabMarkedList = []
    for i in range(len(smsWordsList)):
        vocabMarked = setOfWordsToVecTor(vocabularyList, smsWordsList[i])
        vocabMarkedList.append(vocabMarked)
    return vocabMarkedList

# 通过词库和词向量计算P(S)、P(Wi|S) 、P(Wi|H)
def trainingNaiveBayes(trainMarkedWords, trainCategory):

    numTrainDoc = len(trainMarkedWords)
    numWords = len(trainMarkedWords[0])

    # pSpam是垃圾邮件的先验概率P(S)
    pSpam = sum(trainCategory) / float(numTrainDoc)

    # 统计语料库中词汇在S和H中出现的次数
    # 为解决某个概率为0导致最后乘积为0以及连乘下溢出的问题，将所有词汇初始化出现的次数为1，并将分母初始化为2。
    wordsInSpamNum = np.ones(numWords)
    wordsInHealthNum = np.ones(numWords)
    spamWordsNum = 2.0
    healthWordsNum = 2.0
    for i in range(0, numTrainDoc):
        # 如果是垃圾SMS或邮件
        if trainCategory[i] == 1:
            # wordsInSpamNum是列表，里面存储的是词库中每一个单词出现的次数
            wordsInSpamNum += trainMarkedWords[i]
            # 统计Spam中词汇出现的总次数
            spamWordsNum += sum(trainMarkedWords[i])
        else:
            wordsInHealthNum += trainMarkedWords[i]
            healthWordsNum += sum(trainMarkedWords[i])

    # wordsInSpamNum是列表，里面存储的是语料库中每一个单词出现的次数，
    # 除以Spam或health中词汇出现的总次数，就可以得到每一个单词在S和H中出现的几率
    pWordsSpamicity = np.log(wordsInSpamNum / spamWordsNum)
    pWordsHealthy = np.log(wordsInHealthNum / healthWordsNum)

    return pWordsSpamicity, pWordsHealthy, pSpam

# 获取训练的模型信息
def getTrainedModelInfo():
    # 加载训练获取的语料库信息
    vocabularyList = getVocabularyList('vocabularyList.txt')
    pWordsHealthy = np.loadtxt('pWordsHealthy.txt', delimiter='\t')
    pWordsSpamicity = np.loadtxt('pWordsSpamicity.txt', delimiter='\t')
    fr = open('pSpam.txt')
    pSpam = float(fr.readline().strip())
    fr.close()

    return vocabularyList, pWordsSpamicity, pWordsHealthy, pSpam


# 计算联合概率进行分类
def classify(vocabularyList, pWordsSpamicity, pWordsHealthy, pSpam, testWords):
    testWordsCount = setOfWordsToVecTor(vocabularyList, testWords)
    testWordsMarkedArray = np.array(testWordsCount)
    # 计算P(S|Wi)和P(H|Wi)。要计算P(S|Wi)和P(H|Wi)只需计算P(Wi|S)P(Ci)和(Wi|H)P(Ci)
    # pWordsSpamicity是S中每个单词出现的概率，testWordsMarkedArray是每封邮件中各单词出现的次数。
    p1 = sum(testWordsMarkedArray * pWordsSpamicity) + np.log(pSpam)
    p0 = sum(testWordsMarkedArray * pWordsHealthy) + np.log(1 - pSpam)
    if p1 > p0:
        return 1
    else:
        return 0
