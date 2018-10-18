# coding=utf-8

import pyltp
import os
import argparse
import codecs

class Tool(object):
    def __init__(self, *args, **kwargs):
        self.__LTP_DATA_DIR = 'D:\\NLP\\ltp_data'
        self.__cws_model_path = os.path.join(self.__LTP_DATA_DIR, 'cws.model')
        self.__pos_model_path = os.path.join(self.__LTP_DATA_DIR, 'pos.model')
        self.__par_model_path = os.path.join(self.__LTP_DATA_DIR, 'parser.model')

        self.segmentor = pyltp.Segmentor()
        self.segmentor.load_with_lexicon(self.__cws_model_path, './../data/word_dict.txt')
        self.postagger = pyltp.Postagger()
        self.postagger.load(self.__pos_model_path)
        self.parser = pyltp.Parser()
        self.parser.load(self.__par_model_path)
        
        self.tags_dict = {}

    def load_tag_dict(self, filename):
        with codecs.open(filename, 'r', 'utf-8') as file:
            for line in file:
                words = line.split(' ')
                if line.strip() and len(words) == 2:
                    self.tags_dict[words[0]] = words[1]

    def release(self):
        self.segmentor.release()
        self.postagger.release()
        self.parser.release()

    def __del__(self):
        self.release()

    def cut(self, string):
        words = list(self.segmentor.segment(string))
        return words
    
    def isforeignword(self, string):
        flag = True
        for ch in string:
            if 'a' <= ch <= 'z' or 'A' <= ch <= 'Z':
                pass
            else:
                flag = False
                break
        return flag

    def postag(self, words):
        words_n = []
        if isinstance(words, list):
            words_n = words
            postags = list(self.postagger.postag(words_n))
        elif isinstance(words, str):
            words_n = self.cut(words)
            postags = list(self.postagger.postag(words_n))
        
        for i in range(len(words_n)):
            if self.isforeignword(words_n[i]):
                postags[i] = 'n'
            if self.tags_dict.get(words_n[i]):
                postags[i] = self.tags_dict[words_n[i]]
        return postags
    
    def parse(self, words, *args):
        if isinstance(words, str):
            arcs = self.parser.parse(self.cut(words), self.postag(words))
        elif isinstance(words, list) and args.count() == 1:
            arcs = self.parser.parse(words, args[0])
        return arcs


class Node(object):
    def __init__(self, word='', pos='', num=-1):
        self.word = word
        self.pos = pos
        self.num = num
    
    def  __str__(self):
        return "{}:{}/{}".format(self.num, self.word, self.pos)


class Nodes(object):
    def __init__(self, *args, **kwargs):
        self.nodes = []

    def get_node_by_num(num):
        pass
    
    def add(self, node):
        self.nodes.append(node)
    
    def insert(self, num, node):
        self.nodes.insert(num, node)
    
    def __getitem__(self, num):
        re = Node()
        for node in self.nodes:
            if node.num == num:
                re = node
        return re


class Relation(object):
    def __init__(self, current_node, next_node, relation=''):
        self.current_node = current_node
        self.next_node = next_node
        self.relation = relation

    def __str__(self):
        return "{} --{}->> {}".format(self.current_node, self.relation, self.next_node)


class Relations(object):
    def __init__(self, *args, **kwargs):
        self.relations = []

    def add(self, relation):
        self.relations.append(relation)

    def insert(self, num, relation):
        self.relations.insert(num, relation)

    def get_rlss_by_current_node_num(self, num):
        re = []
        for rls in self.relations:
            if rls.current_node.num == num:
                re.append(rls)
        return re

    def get_rlss_by_next_node_num(self, num):
        re = []
        for rls in self.relations:
            if rls.next_node.num == num:
                re.append(rls)
        return re

    def get_prev_rls_by_num_and_poss(self, current_node_num, prev_pos):
        """
        get the previous rls which has a postag=prev_pos
        """
        rlss = []
        for rls in self.relations:
            if rls.next_node.num == current_node_num and rls.current_node.pos in prev_pos:
                rlss.append(rls)
        return rlss

    def get_rls_by_pos(prev_pos, next_pos):
        re = []
        for rls in self.relations:
            if rls.current_node.pos == prev_pos and rls.next_node.pos == next_pos:
                re.append(rls)
        return re

    """
    def get_noun_sequence(self):
        noun_seq = []
        taboo_num_set = set()
        for rls in self.relations:
            noun_rlss = []
            if rls.current_node.pos in ['n', 'nd', 'nh', 'ni', 'nl', 'ns', 'nt', 'r']:
                if rls.current_node.num not in taboo_num_set:
                    while (rls.next_node.num != -1):
                        noun_rlss.append(rls)
                        taboo_num_set.add(rls.current_node.num)
                        rls_n = self.get_rls_by_current_node_num(rls.next_node.num)

                        rls_n_d = self.get_prev_rls_by_num_and_poss(rls_n.current_node.num, ['d'])
                        # print("node = {}, next = {}, rls = {}, rls_n = {}, rls_n_d = {}".format(rls.current_node, rls.next_node, rls, rls_n, rls_n_d))
                        if rls_n.current_node.pos == 'a' and rls_n_d:
                            noun_rlss.extend(rls_n_d) # 贵前面加`不`, 找到对形容词修饰的副词
                        if rls_n:
                            rls = rls_n
                        else:
                            break
            if noun_rlss:
                noun_seq.append(noun_rlss)
        return noun_seq
        """
    
    def get_seq(self):
        pass
                        

def main(args):
    if args.input:
        string = args.input
    else:
        raise argparse.ArgumentError
    
    # string = args # debug 模式下开启

    tool = Tool()
    tool.load_tag_dict('./../data/tag_dict.txt')
    # tool.load_tag_dict('./data/tag_dict.txt')  # debug模式下开启  
    words = tool.cut(string)
    postags = tool.postag(string)
    arcs = tool.parse(string)

    # print("words = ", words)
    # print("postags = ", postags)
    # print("arcs = " + "  ".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))
    
    rlss = Relations()
    nodes = Nodes()
    
    i = 0
    for arc in arcs:
        current_node = Node(word=words[arc.head - 1], pos=postags[arc.head - 1], num=arc.head - 1)
        next_node = Node(word=words[i], pos=postags[i], num=i)

        nodes.add(current_node)
        nodes.add(next_node)

        # rls = Relation(current_node, next_node, arc.relation)
        rls = Relation(current_node, next_node, arc.relation)        
        rlss.add(rls)

        print(rls)
        i += 1

    print('-'*40)

    # seq = rlss.get_noun_sequence()
    # for _ in seq:
    #     for __ in _:
    #         print(__, end=' ')
    #     print()


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument("input", help="input string u want to analyse")
    args = argp.parse_args()
    # args = '森林人的价格很贵' # debug模式下开启
    main(args)
