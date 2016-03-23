# NaiveBayesSpamFilter V1.0
利用朴素贝叶斯算法实现垃圾邮件的过滤，并结合Adaboost改进该算法。
##1 Naive Bayes spam filtering
&emsp;&emsp;假设邮件的内容中包含的词汇为Wi，垃圾邮件Spam，正常邮件ham。
判断一份邮件，内容包含的词汇为Wi，判断该邮件是否是垃圾邮件，即计算P（S|Wi）这个条件概率。根据Bayes' theorem：

![Bayes' theorem](https://upload.wikimedia.org/math/a/6/e/a6e7f8c521dcf018b6480a8967773ac3.png)

&emsp;&emsp;其中：

- Pr(S|Wi) 出现词汇Wi的邮件是垃圾邮件的条件概率；
- Pr(S)    训练阶段邮件数据集中垃圾邮件的概率，或实际调查的垃圾邮件的概率；
- Pr(Wi|S) 垃圾邮件中词汇Wi出现的概率；
- Pr(H)    训练阶段邮件数据集中正常邮件的概率，或实际调查的正常邮件的概率；
- Pr(Wi|H) 正常邮件中词汇Wi出现的概率；

&emsp;&emsp;对于邮件中出现的所有词汇，考虑每个词汇出现事件的独立性，计算Pr(S|Wi)的联合概率Pr(S|W)，W={W1，W2，...Wn}：

![Bayes' theorem](https://upload.wikimedia.org/math/f/1/d/f1d1c65ee72c294f1fc9b4eb156f5768.png)

&emsp;&emsp;其中：
- P        即Pr(S|W)，出现词汇W={W1，W2......Wn}的邮件是垃圾邮件的条件概率；
- Pi       即Pr(S|Wi)，出现词汇Wi的邮件是垃圾邮件的条件概率；

&emsp;&emsp;**注：**程序中，通过计算出Pr(S|W)和Pr(H|W)，比较Pr(S|W)和Pr(H|W)的大小，判断是垃圾邮件还是正常邮件。我们发现Pr(S|W)和Pr(H|W)计算的分母相同，所以我们只需要比较分子即可。但存在两个问题：

1. 当词汇不存在时，即ni=0，此时Pr(S|Wi) = 0，会造成P=0，无法比较
2. 当Pr(S|Wi)较小时，连乘操作会造成下溢出问题

&emsp;&emsp;**解决方案：**
1. 计算P(Wi|S)和P(Wi|H)时，将所有词汇初始化出现的次数为1，并将分母初始化为2。
```
    # 统计语料库中词汇在S和H中出现的次数
    wordsInSpamNum = np.ones(numWords)
    wordsInHealthNum = np.ones(numWords)
    spamWordsNum = 2.0
    healthWordsNum = 2.0
```
2. 计算P(Wi|S)和P(Wi|H)时，对概率取对数
```
    pWordsSpamicity = np.log(wordsInSpamNum / spamWordsNum)
    pWordsHealthy = np.log(wordsInHealthNum / healthWordsNum)
```




