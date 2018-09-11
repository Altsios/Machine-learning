# -*- coding: utf-8 -*-
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn_deltatfidf import DeltaTfidfVectorizer
from re import sub
from joblib import Parallel, delayed

# стеммеры
eng_stem = SnowballStemmer("english")
rus_stem = SnowballStemmer("russian")

# обработанные стеммером стоп-слова
eng_stw = list(map(eng_stem.stem, set(stopwords.words("english"))))
rus_stw = list(map(rus_stem.stem, set(stopwords.words("russian"))))

# типы н-грамм
ngram = {"BOW": (1, 1), "BG": (2, 2), "TG": (3, 3),
         "BOW_BG": (1, 2), "BOW_BG_TG": (1, 3), "BG_TG": (2, 3)}

# базовые параметры векторизаторов
par = {"lowercase": False,
       "tokenizer": lambda x: x.split(" ")}


# генератор векторизатора
def GetVectorizer(metrics, type,features=5000):
    if metrics == "TF":
        return CountVectorizer(**par,max_features=features, ngram_range=ngram[type])
    elif metrics == "TP":
        return CountVectorizer(**par,max_features=features, ngram_range=ngram[type], binary=True)
    elif metrics == "TFIDF":
        return TfidfVectorizer(**par,max_features=features, ngram_range=ngram[type])
    else:
        return DeltaTfidfVectorizer(**par,max_features=features, ngram_range=ngram[type])

def clean_str(str):

    str=sub("(<[^>]+>|http.*/.*[^\s]|[^\w:\(\);3<^*]|[0-9[^3])",
            " ",str).lower()
    str=sub("(:|;)(.)", " \g<1>\g<2> ",str)
    str=sub("([^:;Xx])([)*(])([^:;Xx]|$)", "\g<1> \g<2> \g<3>",str)
    str=sub("( : | ; | \) | \( | * )", " ",str)
    str=sub(r"(.)"+r"\1{2,}", "\g<1>"*2, str)

    return str

# функция нормализации для всех языков
def normaline(line, stemmer, stops):
    return " ".join(filter(lambda word: word not in stops, map(stemmer.stem, line.split())))

def tokenize(x):
    return x.split(" ")

def normalization(data,lang):

    # для каждого языка свои стоп слова и наилучшая нормализация
    if lang == "русский":

        data = Parallel(n_jobs=-1, backend="multiprocessing")\
            (delayed(normaline)(x, rus_stem, rus_stw) for x in data)

    elif lang == "английский":

        data = Parallel(n_jobs=-1, backend="multiprocessing")\
            (delayed(normaline)(x, eng_stem, eng_stw) for x in data)

    return data

# N-Gram
class Ngram(object):
    def __init__(self, reviews, message_callback,sentiments=None, metrics=None, type=None,
                 vectorizFunc=None,model=None,features=5000):
        self.type = type
        self.metrics = metrics
        self.vectors = vectorizFunc(reviews,model,features,message_callback) if vectorizFunc is not None\
            else self.vectorizer(reviews,sentiments,features,message_callback)

    def vectorizer(self, data, sentiment,features,message_callback):

        # векторизация [TF, TP, TF-IDF, deltaTF-IDF]/[BOW, Bigram, Trigram и их комбинации]
        message_callback.emit("Векторизация...")
        vectorizer = GetVectorizer(self.metrics, self.type,features)
        features = vectorizer.fit_transform(data, sentiment).toarray()
        return features
