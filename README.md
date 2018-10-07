# CCF BDCI 2018 汽车行业用户观点主题及情感识别 比赛试水

--------
<!-- TOC -->

- [CCF BDCI 2018 汽车行业用户观点主题及情感识别 比赛试水](#ccf-bdci-2018-汽车行业用户观点主题及情感识别-比赛试水)
    - [文件目录](#文件目录)
    - [数据分析](#数据分析)
    - [解决方案](#解决方案)
        - [方案一：文本分类法](#方案一文本分类法)
    - [文献和参考资料](#文献和参考资料)

<!-- /TOC -->

---------

## 文件目录
```
│  README.md 说明文档
│
├─data
│  │  hlt_stop_words.txt 停用词表
│  │  test_data_after_cut.pkl content字段已切词的训练集数据
│  │  train_data_after_cut.pkl content字段已切词的测试集数据
│  │
│  ├─analysis
│  │      dataset_analysis.ipynb 数据集测试及初步分析代码
│  │
│  ├─test_public
│  │      test_public.csv 数据集中的测试集文件
│  │
│  └─train
│          train.csv 数据集中的训练集文件
│
├─output
│      output.csv 一个输出样本
│
├─report
│  └─ZhuTong
│      │  ExperimentReport-ZhuTong.md 实验报告
│      │
│      └─figures
│              textinception_f1.jpg
│              textinception_loss.jpg
│
└─src
        SGD.ipynb 使用SGDClassifier进行分类
        SVC.ipynb 使用SVC进行分类
        text_inception.py 一个keras实现的TextInception模型
        word2vec_trainning.py Word2Vec训练代码
```

## 数据分析

详见`data/analysis`

## 解决方案

### 方案一：文本分类法

由于题目中对情感分析的情感词部分并不做强制性要求，故可简单地把本问题看作为两个文本分类问题。

第一个文本分类问题是关于主题的分类；第二个文本分类问题是关于情感极性的分类。

分别进行分类后得到相应的结果，再将结果拼接即可。

**优点：**

- 分类速度快
- 算法复杂度低，开发周期短

**缺点：**

- **分类效果差** 
- 对于特定问题缺乏灵活性
- 参数选取繁琐

目前使用了两个分类器进行分类测试，值得注意的是，目前还未对sentiment_word字段进行有效的应用，且主题分类器和情感分类器之间彼此割裂，分别独立。目前需要解决数据的利用问题和主题与情感割裂的问题。

线上结果如下：
- SVC（RBF核的SVM算法），最终线上测试结果为：`0.35952064000`
- SGD（LinearSVM），最终线上测试结果为：`0.59387480000`

问题解决流程：
1. 文本分词
2. 统计词频
3. 计算TF-IDF矩阵
4. 对上述步骤所产生的矩阵进行分类操作，构建分类器
5. 带入测试集，得到预测结果

## 文献和参考资料

- 分类工具的使用
   - [libSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)
   - [MaxEnt](https://homepages.inf.ed.ac.uk/lzhang10/maxent.html)
   - [sci-kit learn](http://scikit-learn.org/stable/)
   
- 样本不均衡问题
   - 南大一位博士写的综述（比较简单）[PDF](http://lamda.nju.edu.cn/liyf/dm14/111220005.pdf)

- 帮助较大的一些Github仓库
    - [IMDB情感分析-LSTM-tflearn](https://github.com/llSourcell/How_to_do_Sentiment_Analysis)
    - [京东文本分类-SGD-sklearn](https://github.com/stevewyl/keras_text_classification/blob/master/jd/ml_text_classification.py)