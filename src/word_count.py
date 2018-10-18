#coding=utf-8

"""
说明:
    本文件用于统计每个主题的单词和词频并输出, 用作查看与该主题匹配的单词.

参数:
    -s --subject    主题
    -o --output     输出目录(带.xlsx后缀)
    -t --stop       使用停词表
    -v --verbose    输出中间细节

使用方法(cmd): 
    .\word_count.py -s 外观 -o ./../data/word_count/外观.xlsx

可选择的主题有: ['价格', '内饰', '动力', '外观', '安全性', '操控', '油耗', '空间', '舒适性', '配置']

注意事项:
    - 请在输出之前先创建目标文件夹
"""

import pandas as pd
from collections import Counter
import jieba
import codecs
import sys, argparse
import os


def get_stop_word_list(filename="../data/hlt_stop_words.txt"):
    """
    返回停词表
    :param filename: 停词表位置
    :return: <List> 停词表
    """
    stop_words = []
    with codecs.open(filename, "r", "utf=8") as stop_word_file:
        for line in stop_word_file:
            stop_words.append(line.strip())
    return stop_words

def cut(string, stop_words=None):
    words = list(jieba.cut(string.strip()))
    words_return = []
    for word in words:
        if word.strip():
            if stop_words:
                if word.strip() not in stop_words:
                    words_return.append(word.strip())
                else:
                    pass
            else:
                words_return.append(word.strip())
    return words_return

if __name__ == '__main__':
    opt_parser = argparse.ArgumentParser()
    opt_parser.add_argument('-s', '--subject', type=str, required=True, help='subject name')
    opt_parser.add_argument('-o', '--output', type=str, required=True, help='output filename')
    opt_parser.add_argument('-t', '--stop', help='load stopwords')
    opt_parser.add_argument('-v', '--verbose', help='echo details')    
    args = opt_parser.parse_args()

    train_data = pd.read_csv('./../data/train/train.csv')
    stop_words = []
    if args.stop:
        stop_words = get_stop_word_list()

    subjects = ['价格', '内饰', '动力', '外观', '安全性', '操控', '油耗', '空间', '舒适性', '配置']
    for subject in subjects:
        jieba.suggest_freq(subject, True)
    jieba.load_userdict('./../data/word_dict.txt')

    if args.verbose:
        print("------------------------------ cut words ----------------------------------")
    price_words = []
    for line in train_data.loc[train_data["subject"] == args.subject, "content"]:
        price_words.extend(cut(line.strip(), stop_words=stop_words))
    if args.verbose:
        print("------------------------------ count stat ----------------------------------")
    price = Counter(price_words)
    price_sorted = sorted(price.items(), key=lambda x: x[1], reverse=True)
    # print(price_sorted[:50])
    price_df = pd.DataFrame(price_sorted, columns=['word', 'count'])
    price_df.to_excel(args.output, index=False)
