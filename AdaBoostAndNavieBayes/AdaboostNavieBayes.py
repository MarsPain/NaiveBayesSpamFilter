#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_

"""
@Author: MarkLiu
"""
import numpy as np


def textParser(text):
    """
    对SMS预处理，去除空字符串，并统一小写
    :param text:
    :return:
    """
    import re
    regEx = re.compile(r'[^a-zA-Z]|\d')  # 匹配非字母或者数字，即去掉非字母非数字，只留下单词
    words = regEx.split(text)
    # 去除空字符串，并统一小写
    words = [word.lower() for word in words if len(word) > 0]
    return words


def loadSMSData(fileName):
    """
    加载SMS数据
    :param fileName:
    :return:
    """
    f = open(fileName)
    classCategory = []  # 类别标签，1表示是垃圾SMS，0表示正常SMS
    smsWords = []
    for line in f.readlines():
        linedatas = line.strip().split('\t')
        if linedatas[0] == 'ham':
            classCategory.append(0)
        elif linedatas[0] == 'spam':
            classCategory.append(1)
        # 切分文本
        words = textParser(linedatas[1])
        smsWords.append(words)
    return smsWords, classCategory


def createVocabularyList(smsWords):
    """
    创建语料库
    :param smsWords:
    :return:
    """
    vocabularySet = set([])
    for words in smsWords:
        vocabularySet = vocabularySet | set(words)
    vocabularyList = list(vocabularySet)
    return vocabularyList


def getVocabularyList(fileName):
    """
    从词汇列表文件中获取语料库
    :param fileName:
    :return:
    """
    fr = open(fileName)
    vocabularyList = fr.readline().strip().split('\t')
    fr.close()
    return vocabularyList


def setOfWordsToVecTor(vocabularyList, smsWords):
    """
    SMS内容匹配预料库，标记预料库的词汇出现的次数
    :param vocabularyList:
    :param smsWords:
    :return:
    """
    vocabMarked = [0] * len(vocabularyList)
    for smsWord in smsWords:
        if smsWord in vocabularyList:
            vocabMarked[vocabularyList.index(smsWord)] += 1
    return np.array(vocabMarked)


def setOfWordsListToVecTor(vocabularyList, smsWordsList):
    """
    将文本数据的二维数组标记
    :param vocabularyList:
    :param smsWordsList:
    :return:
    """
    vocabMarkedList = []
    for i in range(len(smsWordsList)):
        vocabMarked = setOfWordsToVecTor(vocabularyList, smsWordsList[i])
        vocabMarkedList.append(vocabMarked)
    return vocabMarkedList


def trainingNaiveBayes(trainMarkedWords, trainCategory):
    """
    训练数据集中获取语料库中词汇的spamicity：P（Wi|S）
    :param trainMarkedWords: 按照语料库标记的数据，二维数组
    :param trainCategory:
    :return:
    """
    numTrainDoc = len(trainMarkedWords)
    numWords = len(trainMarkedWords[0])
    # 是垃圾邮件的先验概率P(S)
    pSpam = sum(trainCategory) / float(numTrainDoc)

    # 统计语料库中词汇在S和H中出现的次数
    wordsInSpamNum = np.ones(numWords)
    wordsInHealthNum = np.ones(numWords)
    spamWordsNum = 2.0
    healthWordsNum = 2.0
    for i in range(0, numTrainDoc):
        if trainCategory[i] == 1:  # 如果是垃圾SMS或邮件
            wordsInSpamNum += trainMarkedWords[i]
            spamWordsNum += sum(trainMarkedWords[i])  # 统计Spam中语料库中词汇出现的总次数
        else:
            wordsInHealthNum += trainMarkedWords[i]
            healthWordsNum += sum(trainMarkedWords[i])

    pWordsSpamicity = np.log(wordsInSpamNum / spamWordsNum)
    pWordsHealthy = np.log(wordsInHealthNum / healthWordsNum)

    return pWordsSpamicity, pWordsHealthy, pSpam


def getTrainedModelInfo():
    """
    获取训练的模型信息
    :return:
    """
    # 加载训练获取的语料库信息
    vocabularyList = getVocabularyList('vocabularyList.txt')
    pWordsHealthy = np.loadtxt('pWordsHealthy.txt', delimiter='\t')
    pWordsSpamicity = np.loadtxt('pWordsSpamicity.txt', delimiter='\t')
    fr = open('pSpam.txt')
    pSpam = float(fr.readline().strip())
    fr.close()

    return vocabularyList, pWordsSpamicity, pWordsHealthy, pSpam


def classify(pWordsSpamicity, pWordsHealthy, DS, pSpam, testWordsMarkedArray):
    """
    计算联合概率进行分类
    :param testWordsMarkedArray:
    :param pWordsSpamicity:
    :param pWordsHealthy:
    :param DS:  adaboost算法额外增加的权重系数
    :param pSpam:
    :return:
    """
    # 计算P(Ci|W)，W为向量。P(Ci|W)只需计算P(W|Ci)P(Ci)
    ps = sum(testWordsMarkedArray * pWordsSpamicity * DS) + np.log(pSpam)
    ph = sum(testWordsMarkedArray * pWordsHealthy) + np.log(1 - pSpam)
    if ps > ph:
        return ps, ph, 1
    else:
        return ps, ph, 0
