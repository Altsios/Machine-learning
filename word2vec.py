# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import logging
from gensim.models import Word2Vec
import multiprocessing
import numpy as np

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def createW2VModel(documents,name, size):


    model = Word2Vec(documents,
                                       size=size,  # размер вектора
                                       sg=1,  # используем skip-gram
                                       hs=0,  # не используем софт-макс
                                       window=5,
                                       iter=10,
                                       min_count=2,
                                       sorted_vocab=1,  # слова в словаре сортируются в порядке убывания частот
                                       workers=multiprocessing.cpu_count())

    model.init_sims(replace=True)
    model.wv.save_word2vec_format("{}".format(name))

def GetFeatures(line,model,dim):
    featvect=np.zeros((dim),dtype="float32")
    tf=0

    for word in line:
        if word in model.wv.vocab.keys():

            tf+=1
            # нашли слово в словаре W2V->добавляем значение вектора
            featvect=np.add(featvect,model.wv[word] )

    if tf!=0:
        featvect=np.divide(featvect,tf )

    return featvect


def AverVectorization(reviews, model,dim=300):

    # матрица признаков
    vectRevs=np.zeros((len(reviews),dim),dtype="float32")

    #self.ui.output.append("Векторизация...")
    for i,rew in enumerate(reviews):
        vectRevs[i]=GetFeatures(rew,model,dim)

    return vectRevs