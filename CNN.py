# -*- coding: utf-8 -*-
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D, \
    Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.models import Model
from sklearn.model_selection import train_test_split
import numpy as np
from gensim.models import KeyedVectors
from multiprocessing import Manager, Process
from joblib import Parallel, delayed
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import warnings
from keras import regularizers
import pickle
import keras.backend as K
import json

warnings.simplefilter("ignore")


def recall(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    pp = K.sum(K.round(K.clip(y_true, 0, 1)))
    rec = tp / (pp + K.epsilon())
    return rec


def precision(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    pr_p = K.sum(K.round(K.clip(y_pred, 0, 1)))
    prec = tp / (pr_p + K.epsilon())
    return prec


def f1(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * ((prec * rec) / (prec + rec + K.epsilon()))


def loader(lst, name,message_callback):
    model = KeyedVectors.load_word2vec_format(name)
    lst.append(model)
    message_callback.emit("Модель {} загружена".format(name))


def txtProccessing(data, feat, num,message_callback):
    # получаем уникальные слова, которые будут составлять веса матрицы
    # в виде последовательности "слово":номер, это будет связь с матрицей
    message_callback.emit("    Токенизация...")
    t = Tokenizer(filters="", num_words=feat)
    t.fit_on_texts(data)
    t.word_index = {e: i for e, i in t.word_index.items() if i <= feat}
    wdx = t.word_index
    message_callback.emit("    Найдено уникальных токенов: {}".format(len(wdx)))

    message_callback.emit("    Кодирование текста числами...")
    seq = t.texts_to_sequences(data)
    # длина определяется большим предложением
    length = int(round(sum(Parallel(n_jobs=-1,
                                    backend="multiprocessing")
                           (delayed(len)(x) for x in data)) / len(data)))

    message_callback.emit("    Приведение к одинаковой длине...")
    data = pad_sequences(seq, length)

    # сохраняем словарь связи с матрицей
    with open("./NN/InnerDict/wdx{}.json".format(num), "w", encoding="utf-8") as file:
        json.dump(wdx, file)

    return data, length, wdx

def Convertion(y,classes):

    newy = np.zeros((len(y), classes))

    for i, el in enumerate(y):
        newy[i][el] = 1

    return newy

def MakeEmbMtx(wdx, model, dim):
    emb_mtx = np.zeros((len(wdx) + 1, dim), dtype="float32")

    # кодирование векторами из словаря
    for w, i in wdx.items():

        # если не нашлось слово, остается 0
        if w in model.wv.vocab.keys():
            emb_mtx[i] = model.wv[w]

    return emb_mtx


class CNNSent(object):

    def __init__(self, filter_h=[3, 4, 5],
                 filt_num=100,
                 dropout=0.5,
                 l2=3,
                 batches=50):

        self.filter_h = filter_h
        self.filt_num = filt_num
        self.dropout = dropout
        self.l2 = l2
        self.batches = batches

    def CreateAndTrainModel(self, message_callback,reviews, sentiments,
                            epochs, features, w2v,
                            emb_dim, tr, num,
                            tab,model=None, conn=None
                            ,classes=2,save=True,w2vname=None):

        # Поток для загрузки W2v

        thrW2V = None

        if model is None and w2v:
            model = Manager().list()
            thrW2V = Process(target=loader, args=[model,w2vname,message_callback])
            thrW2V.start()

        message_callback.emit("Создание модели сети...\n  Обработка текста...")
        reviews, lenSent, wdx = txtProccessing(reviews, features, num,message_callback)

        # преборазование для возможности многоклассвоой классификации
        if classes>2:
            sentiments=Convertion(sentiments)

        message_callback.emit("  Разбиение на валидационную и обучающую выборки...")
        x_train, x_val, y_train, y_val = train_test_split(reviews, sentiments,
                                                          test_size=0.3, random_state=13, stratify=sentiments)

        message_callback.emit("    Размер обучающей выборки {}:".format(len(x_train)))
        message_callback.emit("    Размер валидационной выборки {}:".format(len(x_val)))

        message_callback.emit("    Сохранение тестовой выборки в каталоге...")

        with open('./NN/Test/testR{}'.format(num), 'wb') as file_pi:
            pickle.dump(x_val, file_pi)

        with open('./NN/Test/testS{}'.format(num), 'wb') as file_pi:
            pickle.dump(y_val, file_pi)

        emb_mtx = None
        if w2v:
            thrW2V.join()

            emb_dim = model[0].vector_size
            message_callback.emit("  Создание матрицы весов из word2vec...")
            emb_mtx = MakeEmbMtx(wdx, model[0], emb_dim)

        message_callback.emit("  Создание слоев сети...")

        # инициализация тензора
        message_callback.emit("    Слой векторного представления слов.")
        inp = Input(shape=(lenSent,), dtype='int32')

        if w2v:
            emb = Embedding(input_dim=len(wdx) + 1, output_dim=emb_dim, weights=[emb_mtx],
                            input_length=lenSent, trainable=tr)(inp)
        else:
            emb = Embedding(input_dim=len(wdx) + 1, output_dim=emb_dim,
                            input_length=lenSent, trainable=True)(inp)

        # Меняем форму вывода
        resh = Reshape((lenSent, emb_dim, 1))(emb)

        message_callback.emit("    Сверточный слой.")
        c_0 = Conv2D(self.filt_num, kernel_size=(self.filter_h[0], emb_dim), padding='valid',
                     kernel_initializer='normal', activation='relu', use_bias=True)(resh)
        c_1 = Conv2D(self.filt_num, kernel_size=(self.filter_h[1], emb_dim), padding='valid',
                     kernel_initializer='normal', activation='relu', use_bias=True)(resh)
        c_2 = Conv2D(self.filt_num, kernel_size=(self.filter_h[2], emb_dim), padding='valid',
                     kernel_initializer='normal', activation='relu', use_bias=True)(resh)

        message_callback.emit("    Субдискретизационный слой.")
        # strides определяют 1 число
        mxp_0 = MaxPool2D(pool_size=(lenSent - self.filter_h[0] + 1, 1), strides=(1, 1), padding='valid')(
            c_0)
        mxp_1 = MaxPool2D(pool_size=(lenSent - self.filter_h[1] + 1, 1), strides=(1, 1), padding='valid')(
            c_1)
        mxp_2 = MaxPool2D(pool_size=(lenSent - self.filter_h[2] + 1, 1), strides=(1, 1), padding='valid')(
            c_2)

        message_callback.emit("    Полносвязный слой.")
        c_tns = Concatenate(axis=1)([mxp_0, mxp_1, mxp_2])

        # Преобразование в одномерный вектор
        flt = Flatten()(c_tns)

        dr = Dropout(self.dropout)(flt)
        o = Dense(units=1 if classes==2 else classes, activation='softmax', kernel_regularizer=regularizers.l2(3))(dr)

        # Создание модели
        model = Model(inputs=inp, outputs=o)

        # Сохраняем веса
        chkp = ModelCheckpoint('weights3.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_loss', verbose=1,
                               save_best_only=True, mode='auto')
        adam = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=[f1, precision, recall, 'accuracy'])

        message_callback.emit("Модель создана.")
        message_callback.emit("Обучение модели...")

        eaerst = EarlyStopping(monitor='val_loss')
        history = model.fit(x_train, y_train, batch_size=self.batches,
                            epochs=epochs, verbose=1, callbacks=[chkp, eaerst],
                            validation_data=(x_val, y_val))

        if save:
            # СОхранение истории
            with open('./NN/History/trainHistDic{}'.format(num), 'wb') as file_pi:
                pickle.dump(history.history, file_pi)

            # Сохранение модели в файл
            model_js = model.to_json()
            with open("./NN/Model/myCNN{}.json".format(num), "w") as jsf:
                jsf.write(model_js)

            # сохранение всех весов
            model.save_weights("./NN/Weights/myCNNw{}.h5".format(num))

            # занесение результатов в базу
        params = "filter_h={}, filt_num={}, dropout={}, l2={}, batches={}". \
            format(self.filter_h, self.filt_num, self.dropout, self.l2, self.batches)

        try:

            with conn.cursor() as cur:
                cur.execute("insert into {} values ('{}','{}','{}','{}','{}',{},{},{},{},'{}','{}')".
                            format(tab, num, "Word2Vec" if w2v else "EmbLayer",
                                   "Trainable" if tr or not w2v else "NonTrainable",
                                   "CNN", params,
                                   history.history['val_precision'][len(history.history['val_precision']) - 1],
                                   history.history['val_recall'][len(history.history['val_recall']) - 1],
                                   history.history['val_f1'][len(history.history['val_f1']) - 1],
                                   "./NN/Model/myCNN{}.json".format(num),
                                   "./NN/Weights/myCNNw{}.h5".format(num),
                                   './NN/History/trainHistDic{}'.format(num)))
        except Exception:
            message_callback.emit("Таблица {} не найдена".format(CNNSent))
