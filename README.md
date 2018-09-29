# CCF BDCI 2018 汽车行业用户观点主题及情感识别 比赛试水

[//]: <>(TOC)

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



## 文献和参考资料

1. 分类工具的使用

   1. [libSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)、[MaxEnt](https://homepages.inf.ed.ac.uk/lzhang10/maxent.html)、[sci-kit learn](http://scikit-learn.org/stable/)
   
2. 样本不均衡问题
   1. 南大一位博士写的综述（比较简单）[PDF](http://lamda.nju.edu.cn/liyf/dm14/111220005.pdf)

## TODO

- [ ] 分类工具的使用
- [ ] 解决样本不均衡问题
- [ ] 分类器参数调整
