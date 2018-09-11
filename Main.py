# -*- coding: utf-8 -*-

import sys
from form import *
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem
import psycopg2 as ps
from Ngrams import clean_str
from Ngrams import normalization
from Ngrams import Ngram
from Classifiers import CalcMet
from Reports import *
from word2vec import AverVectorization
from word2vec import createW2VModel
from CNN import CNNSent
from Ngrams import tokenize
from joblib import Parallel, delayed
from multiprocessing import Manager,Process
from CNN import loader
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem
from PyQt5.QtCore import pyqtSlot,pyqtSignal,QObject,QThread


class sgn(QObject):

    compl = pyqtSignal()
    res = pyqtSignal(object)
    message = pyqtSignal(object)


class sgn(QObject):
    compl = pyqtSignal()
    res = pyqtSignal(object)
    message = pyqtSignal(object)


class Worker(QThread):

    def __init__(self, func, *args, **kwargs):
        super(Worker, self).__init__()

        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.sgn = sgn()

        self.kwargs['message_callback'] = self.sgn.message

    @pyqtSlot()
    def run(self):
        res = self.func(*self.args, **self.kwargs)

        self.sgn.res.emit(res)
        self.sgn.compl.emit()

class Table(QTableWidget):
    def __init__(self,df,txt):
        super(Table, self).__init__()
        self.setWindowTitle(txt)
        h = df.columns.values.tolist()
        self.setColumnCount(len(h))
        self.setHorizontalHeaderLabels(h)
        self.setAlternatingRowColors(True)


        for i, row in df.iterrows():
            # Добавление строки
            self.setRowCount(self.rowCount() + 1)

            for j in range(self.columnCount()):
                self.setItem(i, j, QTableWidgetItem(str(row[j])))

        self.resizeColumnsToContents()


class MyApp(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.twdsets.setColumnWidth(0, 105)
        self.ui.twdsets.setColumnWidth(1, 105)
        self.ui.twdsets.setColumnWidth(2, 50)
        self.model = Manager().list()
        # для динамического изменения таблицы
        self.vB = self.ui.twdsets.verticalScrollBar()
        self.vBarllstv = self.vB.value()
        self.worker=None

        self.vB.valueChanged.connect(self.scrCh)

        # действия при нажатии
        self.ui.start.clicked.connect(self.StartThread)
        self.ui.okdropres.clicked.connect(self.StartThread)
        self.ui.okdatanorm.clicked.connect(self.StartThread)
        self.ui.okw2v.clicked.connect(self.StartThread)
        self.ui.show.clicked.connect(self.Reports)

    def msf(self, mess):
        self.ui.output.append(str(mess))

    def Cancelation(self):
        self.worker.terminate()
        self.ui.output.append("Отмена действия.")
        self.enabled()

    def StartThread(self):

        self.disabled()
        exfunc=None

        sender = self.sender().objectName()

        if sender=="start":
            exfunc=self.Start
        elif sender=="okdatanorm":
            exfunc=self.unionAct
        elif sender=="okdropres":
            exfunc=self.DropFromTab
        elif sender=="okw2v":
            exfunc=self.CreateW2Vmodel

        self.worker = Worker(exfunc)
        self.worker.sgn.message.connect(self.msf)
        self.worker.sgn.compl.connect(self.enabled)
        self.worker.start()
        self.ui.cancel.clicked.connect(self.Cancelation)

    def Reports(self):

        try:
            df=None
            text=str(self.ui.comboBox.currentText())

            if text=='Сводная диаграмма качества классификации':
                clasdata, w2vdata, cnndata = finalReport(self.ui.txtconnstr.toPlainText())
                summary(clasdata, w2vdata, cnndata)
                return
            elif text=='График функции потерь':
                showloss(self.ui.losspath.toPlainText())

                return

            elif text=='График F1-меры':
                paths=self.ui.f1path.toPlainText().split(', ')
                showF1gr(paths)

                return


            if text=="Лучшие результаты классического подхода":
                df=BestF1(self.ui.txtconnstr.toPlainText(),
                          self.ui.allresultsname.toPlainText(),
                          'classic')
            elif text=="Лучшие результаты подхода Word2Vec":
                df = BestF1(self.ui.txtconnstr.toPlainText(),
                            self.ui.allresultsname.toPlainText(),
                            'w2v')

            elif text=='Оптимальные параметры':
                df=BestParams(self.ui.txtconnstr.toPlainText(),
                            self.ui.allresultsname.toPlainText())

            elif text=='Топ 5 алгоримтов классифкации':
                df=BestAlg(self.ui.txtconnstr.toPlainText(),
                            self.ui.allresultsname.toPlainText())

            elif text=='Оценка нейронной сети (по выборкам)':
                df=CnnInfo(self.ui.cnnname.toPlainText(),
                           self.ui.txtconnstr.toPlainText())

            elif text=='Оценка нейронной сети (объединенная выборка)':
                df = CnnInfo(self.ui.cnnname.toPlainText(),
                             self.ui.txtconnstr.toPlainText(),
                             onlyAll=True)

            elif text == 'Общий случай (лучшие по объединенной выборке)':
                df=Compare(self.ui.allresultsname.toPlainText(),
                           self.ui.cnnname.toPlainText(),
                           self.ui.txtconnstr.toPlainText())



            self.w2 = Table(df,text)
            self.w2.show()
        except Exception:
            self.ui.output.append("Таблица не найдена или ее структура не соответствует требованиям")


    def CreateW2Vmodel(self,message_callback):

        self.disabled()

        data = self.getAll()[0]
        message_callback.emit("Токенизация...")

        data = Parallel(n_jobs=-1, backend="multiprocessing") \
            (delayed(tokenize)(x) for x in data)

        message_callback.emit("Создание модели...")
        createW2VModel(data, name=self.ui.w2vname.toPlainText(),
                       size=self.ui.w2vdim.toPlainText())  # список таблиц, из которых берутся данные

        self.enabled()

    # Обработка прокрутки таблицы
    def scrCh(self, val):

        rowCount = self.ui.twdsets.rowCount()

        # прокрута вниз
        if val > self.vBarllstv :
            self.ui.twdsets.insertRow(rowCount)

        # прокрутка вверх
        elif val < self.vBarllstv:
            lstR = rowCount - 1
            e = True
            for col in range(self.ui.twdsets.columnCount()):
                item = self.ui.twdsets.item(lstR, col)
                if item != '':
                    e = False
                    break
            if e:
                self.ui.twdsets.removeRow(lstR)

        self.vBarllstv = val

    def unionAct(self,message_callback):

        dtab = self.ui.alldatanorm.toPlainText()
        conn_str= self.ui.txtconnstr.toPlainText()

        try:
            if self.ui.checkBox_14.isChecked():
                message_callback.emit("Удаление из {}".format(dtab))

                with ps.connect(conn_str) as conn:
                    with conn.cursor() as cur:
                        cur.execute("delete from {}".format(dtab))

            if self.ui.checkBox_15.isChecked():
                message_callback.emit("Добавление данных в {}".format(dtab))
                self.fappend(message_callback)

            if not self.ui.checkBox_14.isChecked() and\
                not self.ui.checkBox_15.isChecked():
                message_callback.emit("Действие не выбрано.")
        except Exception:
            message_callback.emit("Таблица {} не найдена в БД".format(dtab))


    def fappend(self,message_callback,query=False):

        DataSets, DataLang = self.GetDsetsnames(message_callback)

        # таблица объединенных выборок
        dtab = self.ui.alldatanorm.toPlainText()
        # строка подключения
        conn_str = self.ui.txtconnstr.toPlainText()

        allreviws=[]

        for table in DataSets.keys():

            with ps.connect(conn_str) as conn:

                with conn.cursor() as cur:

                    message_callback.emit("Загрузка данных из БД...")
                    try:
                        cur.execute("select * from {}".format(DataSets[table]))
                    except Exception:
                        message_callback.emit("Таблицы {} не существует".format(table))
                        continue
                    if cur.rowcount == 0:
                        message_callback.emit("В таблице {} нет данных.".format(table))
                        continue
                    message_callback.emit("Данные из {} получены ({} строк)".format(table, cur.rowcount))

                    reviews = []
                    sent = []

                    message_callback.emit("Преобразование в список...")

                    for row in cur.fetchall():
                        sent.append(int(row[0]))
                        reviews.append(row[1])

                    if self.ui.checkBox_4.isChecked():

                        message_callback.emit("Нормализация...")
                        reviews = normalization(reviews, DataLang[table])

                    allreviws+=reviews

            # нормализация данных
            if self.ui.checkBox_3.isChecked():

                message_callback.emit("Применение регулярных выражений...")
                # удаление лишних символов, повторов. сохранение распространенных "смайликов"
                reviews = Parallel(n_jobs=-1, backend="multiprocessing") \
                    (delayed(clean_str)(x) for x in reviews)

            message_callback.emit("Запись в базу...")
            for i in range(len(allreviws)):
                try:
                    cur.execute("insert into {} values ({},'{}')"
                            .format(dtab, sent[i], allreviws[i]))
                except Exception:
                    message_callback.emit("Таблица {} не найдена в БД".format(dtab))

        if query:
            self.getAll(message_callback)

    def getAll(self,message_callback):

        datasets = []
        sentiments = []

        message_callback.emit("Загрузка данных из БД...")

        # таблица объединенных выборок
        dtab = self.ui.alldatanorm.toPlainText()
        # строка подключения
        conn_str = self.ui.txtconnstr.toPlainText()
        try:

            with ps.connect(conn_str) as conn:

                with conn.cursor() as cur:
                    cur.execute("select * from {}".format(dtab))
                    if cur.rowcount != 0:
                        for row in cur.fetchall():
                            sentiments.append(int(row[0]))
                            datasets.append(row[1])

                    else:
                        self.fappend(query=True)
        except Exception:
            message_callback.emit("Таблица {} не найдена в БД".format(dtab))

        return [datasets, sentiments]

    # сохраняем данные в словаре
    def GetDsetsnames(self,message_callback):

        message_callback.emit("Получение информации о наборах данных...")

        ds={}
        dl={}

        allRows = self.ui.twdsets.rowCount()
        for row in range(allRows):
            e = False
            for col in range(self.ui.twdsets.columnCount()):
                item = self.ui.twdsets.item(row, col)
                if item=='':
                    e=True
                    break
            if e:
                continue
            key=self.ui.twdsets.item(row, 0).text()
            ds[key]=self.ui.twdsets.item(row, 1).text()
            dl[key]=self.ui.twdsets.item(row, 2).text()

        return ds, dl

    def ReadTxt(self):
        # таблица объединенных выборок
        dtab = self.ui.alldatanorm.toPlainText()

        # таблицы с результатами
        tab = self.ui.allresultsname.toPlainText()
        tabCNN = self.ui.cnnname.toPlainText()

        # строка подключения
        conn_str = self.ui.txtconnstr.toPlainText()

        return dtab, tab, tabCNN, conn_str

    # удаление из таблиц
    def DropFromTab(self,message_callback):

        message_callback.emit("Очищение таблиц...")
        dtab,tab,tabCNN,conn_str = self.ReadTxt()


        with ps.connect(conn_str) as conn:
            with conn.cursor() as cur:
                if self.ui.checkBox_12.isChecked():
                    message_callback.emit("Удаление из {}".format(tab))
                    try:
                        cur.execute("delete from {}".format(tab))
                    except Exception:
                        message_callback.emit("Таблица {} не найдена в БД".format(tab))

                if self.ui.checkBox_13.isChecked():
                    message_callback.emit("Удаление из {}".format(tabCNN))
                    try:
                        cur.execute("delete from {}".format(tabCNN))
                    except Exception:
                        message_callback.emit("Таблица {} не найдена в БД".format(tabCNN))

        if not self.ui.checkBox_12.isChecked() and\
                not self.ui.checkBox_13.isChecked():
            message_callback.emit("Действие не выбрано.")

    # классический подход
    def classicF(self,sreviews, sentiments, table,
                 DataLang,Lmetrics,Ltype,cls,conn,feat,message_callback):

        message_callback.emit("Выполняется классический подход...")
        # Генерация всевозможных параметров входных данных
        for metrics, type in [(metrics, type)
                              for metrics in Lmetrics
                              for type in Ltype]:

            if self.ui.checkBox_3.isChecked():
                message_callback.emit("Применение регулярных выражений...")
            # удаление лишних символов, повторов. сохранение распространенных "смайликов"
                sreviews = Parallel(n_jobs=-1, backend="multiprocessing") \
                    (delayed(clean_str)(x) for x in sreviews)

            ngram = Ngram(reviews=normalization(sreviews, DataLang[table])
            if self.ui.checkBox_4.isChecked() and table!="Все" else sreviews,
                          sentiments=sentiments,
                          type=type,
                          metrics=metrics,
                          features=feat,message_callback=message_callback)

            CalcMet(message_callback,table, ngram, sentiments,
                    self.ui.allresultsname.toPlainText(), cls,
                    True if self.ui.checkBox.isChecked() else False,
                    conn)

    # подход w2v
    def w2vf(self,sreviews, sentiments, table, DataLang, cls, conn,message_callback):

        message_callback.emit("Выполняется подход word2vec...")
        message_callback.emit("Загрузка модели word2vec...")

        if self.ui.checkBox_3.isChecked() and table != 'Все':
            message_callback.emit("Применение регулярных выражений...")
        # удаление лишних символов, повторов. сохранение распространенных "смайликов"
            sreviews = Parallel(n_jobs=-1, backend="multiprocessing") \
                (delayed(clean_str)(x) for x in sreviews)

        thrW2V = Process(target=loader, args=[self.model,self.ui.w2vname.toPlainText(),message_callback])
        thrW2V.start()

        sreviews = Parallel(n_jobs=-1, backend="multiprocessing") \
            (delayed(tokenize)(x) for x in normalization(sreviews, DataLang[table]
             if self.ui.checkBox_4.isChecked()  and table!="Все" else sreviews))

        thrW2V.join()

        # представление текстов, как усредненных векторов
        vectorizedtxt = Ngram(reviews=sreviews, message_callback=message_callback,
                              metrics="AveragedVectors",
                              type="Word2Vec", vectorizFunc=AverVectorization,
                              model=self.model[0], features=self.model[0].vector_size)

        CalcMet(message_callback,table, vectorizedtxt, sentiments,
                self.ui.allresultsname.toPlainText(),
                cls,True if self.ui.checkBox.isChecked() else False,conn)

    # сверточная нейронаая сеть
    def CNNf(self,sreviews,sentiments,table,feat,DataLang,message_callback):

        if self.ui.checkBox_3.isChecked() and table!='Все':
            message_callback.emit("Применение регулярных выражений...")
        # удаление лишних символов, повторов. сохранение распространенных "смайликов"
            sreviews = Parallel(n_jobs=-1, backend="multiprocessing") \
                (delayed(clean_str)(x) for x in sreviews)

        sreviews = Parallel(n_jobs=-1, backend="multiprocessing") \
            (delayed(tokenize)(x) for x in normalization(sreviews, DataLang[table]
            if self.ui.checkBox_4.isChecked() and table != "Все" else sreviews))

        message_callback.emit("Инициализация модели сети...")
        # Создаем экземпляр сети
        ConvNN = CNNSent() \
            .CreateAndTrainModel(message_callback=message_callback,
                                 w2vname=self.ui.w2vname.toPlainText(),
                                 epochs=100,
                                 reviews=sreviews,
                                 sentiments=sentiments,
                                 features=feat,
                                 w2v=True if self.ui.w2vvectors.isChecked() else False,
                                 emb_dim=int(self.ui.emb_dim.toPlainText()),
                                 tr=True if self.ui.train.isChecked() else False,
                                 num=table,
                                 model=self.model, conn=self.ui.txtconnstr.toPlainText()
                                 , tab=self.ui.cnnname.toPlainText()
                                 ,classes=int(self.ui.cntclss.toPlainText()),
                                 save=True if self.ui.checkBox_2.isChecked() else False)

    def GetMN(self,message_callback):

        message_callback.emit("Получение информации о метриках и N-граммах...")

        l = []
        n = []

        allRows = self.ui.lstMetrics.count()
        for row in range(allRows):
            item = self.ui.lstMetrics.item(row).text()
            if item == '':
                continue
            l.append(item)

        allRows = self.ui.lstngrams.count()
        for row in range(allRows):
            item = self.ui.lstngrams.item(row).text()
            if item=='':
                continue
            n.append(item)

        return l,n

    def GetCP(self,message_callback):

        message_callback.emit("Получение информации о классификаторах...")

        ds = {}

        allRows = self.ui.twclassif.rowCount()
        for row in range(allRows):
            e = False
            for col in range(self.ui.twclassif.columnCount()):
                item = self.ui.twclassif.item(row, col)
                if item=='':
                    e = True
                    break
            if e:
                continue
            key = self.ui.twclassif.item(row, 0).text()
            ds[key] = self.ui.twclassif.item(row, 1).text()

        return ds

    def disabled(self):

        self.ui.cancel.setEnabled(True)
        self.ui.start.setDisabled(True)
        self.ui.okdropres.setDisabled(True)
        self.ui.okdatanorm.setDisabled(True)
        self.ui.okw2v.setDisabled(True)
        self.ui.show.setDisabled(True)

    def enabled(self):

        self.ui.cancel.setDisabled(True)
        self.ui.start.setEnabled(True)
        self.ui.okdropres.setEnabled(True)
        self.ui.okdatanorm.setEnabled(True)
        self.ui.okw2v.setEnabled(True)
        self.ui.show.setEnabled(True)

    def Start(self,message_callback):

        if not self.ui.checkBox_5.isChecked() and \
        not self.ui.checkBox_6.isChecked() and \
        not self.ui.checkBox_7.isChecked() and \
        not self.ui.checkBox_5.isChecked() and \
        not self.ui.checkBox_6.isChecked() and \
        not self.ui.checkBox_7.isChecked():
            message_callback.emit("Ничего не выбрано...")
            return

        dtab, tab, tabCNN, conn_str = self.ReadTxt()

        # получить имена таблиц
        DataSets, DataLang=self.GetDsetsnames(message_callback)
        # получить метрики
        if self.ui.checkBox_5.isChecked():
            Lmetrics,Ltype= self.GetMN(message_callback)

        if self.ui.checkBox_5.isChecked() or \
                self.ui.checkBox_6.isChecked():
            cls=self.GetCP(message_callback)

        #  если нужны отдельные наборы данных
        if self.ui.checkBox_5.isChecked() or\
           self.ui.checkBox_6.isChecked() or \
           self.ui.checkBox_7.isChecked():
            with ps.connect(conn_str) as conn:
                # выполняем работу для каждого датасета
                for table in DataSets.keys():

                    message_callback.emit("Загрузка данных из БД...")
                    with conn.cursor() as cur:

                        limit=self.ui.limit.toPlainText()
                        sql="select * from {}{}".format(DataSets[table],
                                                         " limit {}".format(limit)\
                                                         if limit!="-1" else "")
                        try:
                            cur.execute(sql.format(DataSets[table]))
                        except Exception:
                            message_callback.emit("Таблицы {} не существует".format(table))
                            continue

                        if cur.rowcount == 0:
                            message_callback.emit("В таблице {} нет данных.".format(table))
                            continue
                        sentiments = []
                        sreviews = []

                        for row in cur.fetchall():
                            sentiments.append(int(row[0]))
                            sreviews.append(row[1])

                    message_callback.emit("Данные из {} получены".format(table))

                    # Классический подход
                    if self.ui.checkBox_5.isChecked():
                        self.classicF(sreviews, sentiments,table,DataLang,Lmetrics,Ltype,cls,conn,
                                      feat=int(self.ui.feateach.toPlainText()),message_callback=message_callback)

                    # Подход word2vec
                    if self.ui.checkBox_6.isChecked():
                        self.w2vf(sreviews, sentiments, table, DataLang, cls, conn,message_callback=message_callback)

                    # сверточная нейронная сеть
                    if self.ui.checkBox_7.isChecked():
                        self.CNNf(sreviews, sentiments, table,DataLang=DataLang,
                                  feat=int(self.ui.feateach.toPlainText()),message_callback=message_callback)

                    data=None
            # Классический подход по объед.
                if self.ui.checkBox_9.isChecked():
                    data = self.getAll(message_callback)
                    self.classicF(data[0], data[1], "Все",DataLang, Lmetrics, Ltype, cls, conn,
                                  feat=int(self.ui.featAll.toPlainText()),message_callback=message_callback)

            # Подход word2vec по объед.
                if self.ui.checkBox_10.isChecked():
                    if data is None:
                        data=self.getAll(message_callback)
                    self.w2vf(data[0], data[1], "Все", DataLang, cls, conn,message_callback=message_callback)

            # сверточная нейронная сеть по объед.
                if self.ui.checkBox_11.isChecked():
                    if data is None:
                        data=self.getAll(message_callback)

                    self.CNNf(data[0], data[1],"Все",DataLang=DataLang,feat=int(self.ui.featAll.toPlainText()),
                              message_callback=message_callback)



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myapp = MyApp()
    myapp.show()
    sys.exit(app.exec_())
