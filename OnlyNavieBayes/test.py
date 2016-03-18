#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_

"""
@Author: MarkLiu
"""

import numpy as np
import OnlyNavieBayes.NaiveBayes as util

pWordsHealthy = np.loadtxt('pWordsHealthy.txt', delimiter='\t')
pWordsSpamicity = np.loadtxt('pWordsSpamicity.txt', delimiter='\t')
print 'pWordsHealthy:', len(pWordsHealthy), np.shape(pWordsHealthy)
print 'pWordsSpamicity:', len(pWordsSpamicity)
pSpam = 0.13401507
pSWi = util.bayesTheoremCalcPSWi(pWordsSpamicity, pWordsHealthy, pSpam)
print pSWi
