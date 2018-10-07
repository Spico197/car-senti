import multiprocessing
from gensim.models import Word2Vec
import pandas as pd

if __name__ == '__main__':
    train_data = pd.read_pickle('./../data/train_data_after_cut.pkl')

    words = []
    for line in train_data.content.values:
        words.append(line.strip().split(" "))

    model1 = Word2Vec(
        words, size=400, window=3,
        min_count=3, workers=multiprocessing.cpu_count()
    )
    model1.save('./../output/model/train_data_vec.model')
    model1.wv.save_word2vec_format('./../output/model/train_data.vector', binary=False)
