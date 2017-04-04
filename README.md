# NaiveBayesSpamFilter
利用朴素贝叶斯算法实现垃圾邮件的过滤，并结合Adaboost改进该算法。

## 1 Naive Bayes spam filtering
&emsp;&emsp;假设邮件的内容中包含的词汇为Wi，垃圾邮件Spam，正常邮件ham。
判断一份邮件，内容包含的词汇为Wi，判断该邮件是否是垃圾邮件，即计算P（S|Wi）这个条件概率。根据Bayes' theorem：

&emsp;&emsp;![Bayes' theorem](https://upload.wikimedia.org/math/a/6/e/a6e7f8c521dcf018b6480a8967773ac3.png)

&emsp;&emsp;其中：

- Pr(S|Wi) 出现词汇Wi的邮件是垃圾邮件的条件概率（即后验概率）；
- Pr(S)    训练阶段邮件数据集中垃圾邮件的概率，或实际调查的垃圾邮件的概率（即先验概率）；
- Pr(Wi|S) 垃圾邮件中词汇Wi出现的概率；
- Pr(H)    训练阶段邮件数据集中正常邮件的概率，或实际调查的正常邮件的概率；
- Pr(Wi|H) 正常邮件中词汇Wi出现的概率；

&emsp;&emsp;对于邮件中出现的所有词汇，考虑每个词汇出现事件的独立性，计算Pr(S|Wi)的联合概率Pr(S|W)，W={W1，W2，...Wn}：

&emsp;&emsp;![Bayes' theorem](https://upload.wikimedia.org/math/f/1/d/f1d1c65ee72c294f1fc9b4eb156f5768.png)

&emsp;&emsp;其中：
- P        即Pr(S|W)，出现词汇W={W1，W2......Wn}的邮件是垃圾邮件的条件概率；
- Pi       即Pr(S|Wi)，出现词汇Wi的邮件是垃圾邮件的条件概率；

&emsp;&emsp;**注：** 程序中，通过计算出Pr(S|W)和Pr(H|W)，比较Pr(S|W)和Pr(H|W)的大小，判断是垃圾邮件还是正常邮件。我们发现Pr(S|W)和Pr(H|W)计算的分母相同，所以我们只需要比较分子即可。

&emsp;&emsp;**但存在两个问题：**

1. 当词汇不存在时，即ni=0，此时Pr(S|Wi) = 0，会造成P=0，无法比较
2. 当Pr(S|Wi)较小时，连乘操作会造成下溢出问题

&emsp;&emsp;**解决方案：**
- 计算P(Wi|S)和P(Wi|H)时，将所有词汇初始化出现的次数为1，并将分母初始化为2（或根据样本/实际调查结果调整分母的值）。
```
    # 统计语料库中词汇在S和H中出现的次数
    wordsInSpamNum = np.ones(numWords)
    wordsInHealthNum = np.ones(numWords)
    spamWordsNum = 2.0
    healthWordsNum = 2.0
```
- 计算P(Wi|S)和P(Wi|H)时，对概率取对数
```
    pWordsSpamicity = np.log(wordsInSpamNum / spamWordsNum)
    pWordsHealthy = np.log(wordsInHealthNum / healthWordsNum)
```
&emsp;&emsp;所以最终比较的是，P(W1|S)P(W2|S)....P(Wn|S)P(S)和P(W1|H)P(W2|H)....P(Wn|H)P(H)的大小。
```
    ps = sum(testWordsMarkedArray * pWordsSpamicity) + np.log(pSpam)
    ph = sum(testWordsMarkedArray * pWordsHealthy) + np.log(1 - pSpam)
```
&emsp;&emsp;**测试效果：** 5574个样本，采用交叉验证，随机选取4574个作为训练样本，产生词汇列表（语料库），对1000个测试样本，分类的平均错误率约为：2.5%。

## 2 Running Adaboost on Naive Bayes

&emsp;&emsp;我们在计算ps和ph联合后验概率时，可引入一个调整因子DS，其作用是调整词汇表中某一词汇的“垃圾程度”(spamicity)，
```
    ps = sum(testWordsMarkedArray * pWordsSpamicity * DS) + np.log(pSpam)
```
&emsp;&emsp;其中DS通过Adaboost算法迭代获取最佳值。原理如下：
```
设定adaboost循环的次数count
交叉验证随机选择1000个样本
DS初始化为和词汇列表大小相等的全一向量
迭代循环count次：
    设定最小分类错误率为inf
    对于每一个样本：
        在当前DS下对样本分类
        如果分类出错：
            计算出错的程度，即比较ps和ph的相差alpha
            如果样本原本是spam，错分成ham：
                DS[样本包含的词汇] = np.abs(DS[样本包含的词汇] - np.exp(alpha) / DS[样本包含的词汇])
            如果样本原本是ham，错分成spam：
                DS[样本包含的词汇] = DS[样本包含的词汇] + np.exp(alpha) / DS[样本包含的词汇]
    计算错误率
    保存最小的错误率和此时的词汇列表、P(Wi|S)和P(Wi|H)、DS等信息，即保存训练好的最佳模型的信息
```
&emsp;&emsp;**测试效果：** 5574个样本，获取Adaboost算法训练的最佳模型信息（包括词汇列表、P(Wi|S)和P(Wi|H)、DS等），对1000个测试样本，分类的平均错误率约为：0.5%。

## References
[Running Adaboost on Naive Bayes](http://web.cecs.pdx.edu/~mm/MachineLearningWinter2010/BoostingNaiveBayes.pdf)<br>
[Boosting and naive bayesian learning](http://pages.cs.wisc.edu/~dyer/cs540/handouts/elkan97boosting.pdf)<br>
[Naive Bayes spam filtering](https://en.wikipedia.org/wiki/Naive_Bayes_spam_filtering)<br>

## License
This project is licensed under the terms of the MIT license.
