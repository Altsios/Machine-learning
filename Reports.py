# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import psycopg2 as ps
import pickle
import os

# лучшие классификаторы
def BestF1(conn_str, tab, type='classic'):

    if type == 'classic':
        pred = " where typen != 'Word2Vec'"

    elif type == "w2v":
        pred = " where typen ='Word2Vec'"
    else:
        pred = ""

    sql = """select setname "Имя набора",typen "N-грамма",metrics "Метрика",classifier "Классификатор",f1 "F1-мера" from(SELECT *, row_number()\
                        OVER (PARTITION BY setname ORDER BY f1 DESC, length(typen))  AS rating\
                        FROM {}{}) ranked where rating = 1""".format(tab, pred)

    with ps.connect(conn_str) as conn:
        data = pd.read_sql(sql, conn)

    return data

def BestParams(conn_str, tab):

    data=None

    sql="""select "Классификатор","Параметры", "Ср F1"
    from(SELECT classifier "Классификатор",param "Параметры", avg(f1) "Ср F1", 
    row_number() over(partition by classifier order by avg(f1) desc) as r
    FROM {} 
    where setname!='All'
    group by classifier,param
    having count(*)>1) t
    where r=1
    """.format(tab)

    with ps.connect(conn_str) as conn:
        data = pd.read_sql(sql, conn)

    return data

#лучший алг-м
def BestAlg(conn_str, tab):

    data = None

    sql = "select allresults.classifier \"Классификатор\", typen \"Вектор\",metrics \"Метрика\", \
       param \"Параметры\", \
            to_char(avg(f1),'FM0.9999') \"Ср F1\" \
                        FROM {} \
                        group by allresults.classifier,typen,metrics,param \
                        order by \"Ср F1\" desc".format(tab)

    with ps.connect(conn_str) as conn:
        df = pd.read_sql(sql, conn)
    return df


def CnnInfo(tab, conn_str, notAll=True, onlyAll=False):

    data = None

    if onlyAll:
        sql = """select typen "N-грамма",metrics "Метрика",f1 "F1-мера",prec "Точность",recall "Полнота" from {} where setname='All'""".format(tab)
    else:
        sql = """select setname "Имя набора",f1 "F1-мера",prec "Точность",recall "Полнота" from {}""".format(tab)

        if notAll:
            sql += " where setname!='All'"

    with ps.connect(conn_str) as conn:
        data = pd.read_sql(sql, conn)
    return data


def Compare(taball, tabcnn, conn_str):

    data = None

    sql = """select classifier "Классификатор",
    typen "Вектор", metrics "Метрика",f1 "F1-мера" from {} where setname='All'
    union all
    select classifier,typen,metrics,f1 from {} where setname='All'
    order by "F1-мера" desc
    """.format(tabcnn, taball)

    with ps.connect(conn_str) as conn:
        data = pd.read_sql(sql, conn)
    return data


def finalReport(conn_str, tab='reports.allresults', cnntab='cnn.allresults', clscol='classifier',
                parcol="param", tcol='typen', mcol='metrics', typen='BOW_BG', metrics='deltaTFIDF',
                ctypen='EmbLayer', cmetrics='Trainable'):
    with ps.connect(conn_str) as conn:
        sql = """select {0} "Классификатор", avg(f1) "Ср F1" 
        FROM {1} a join 
                         (select "Классификатор","Параметры", "Ср F1"
                          from(SELECT {0} "Классификатор",{2} "Параметры", avg(f1) "Ср F1", 
                          row_number() over(partition by {0} order by avg(f1) desc) as r
                          FROM {1} 
                          where setname!='All'
                          group by {0},{2}) c
                          where r=1) b on b."Классификатор"=a.{0} and b."Параметры"=a.{2}
        where {3}='{4}' and {5}='{6}'
        group by {0} 
        order by "Классификатор" desc""".format(clscol, tab, parcol, tcol, typen, mcol, metrics)

        dcl = pd.read_sql(sql, conn)

        sql = sql.replace("where {}='{}' and {}='{}'".format(tcol, typen, mcol, metrics),
                          "where {}='Word2Vec'".format(tcol))
        wcl = pd.read_sql(sql, conn)

        sql = sql.replace("where {}='Word2Vec'".format(tcol),
                          "where typen='{}' and metrics='{}'"
                          .format(ctypen, cmetrics)).replace('reports.allresults', 'reports.cnnresults')

        ccl = pd.read_sql(sql, conn)
        return [dcl, wcl, ccl]

def showloss(path):

    if os.path.exists(path):
        with open (path,"rb") as f:
            history=pickle.load(f)

    else:
        return

    val_loss= history['val_loss']
    loss= history['loss']
    epoch = range(1,len(val_loss)+1)

    plt.figure(figsize=(11, 5))
    plt.xticks(epoch)

    #задаем внешний вид линий
    line0, line1= plt.plot(epoch,val_loss, 'r', epoch,loss, 'b')

    #подписи осей

    plt.xlabel('Эпохи')
    plt.ylabel('Ошибка')

    # легенда
    plt.annotate ('{:.4f}'.format(val_loss[len(val_loss)-2]),
                        xy=(len(val_loss)-1, val_loss[len(val_loss)-2]))
    plt.annotate ('{:.4f}'.format(loss[len(loss)-2]),
                        xy=(len(loss)-1, loss[len(loss)-2]))
    plt.annotate ('{:.4f}'.format(val_loss[len(val_loss)-1]),
                        xy=(len(val_loss), val_loss[len(val_loss)-1]))
    plt.annotate ('{:.4f}'.format(loss[len(loss)-1]),
                        xy=(len(loss), loss[len(loss)-1]))
    plt.legend( (line0, line1), ('Ошибка на тестовом множестве', 'Ошибка на обучающем множестве'), loc = 'best')

    plt.title("График функции потерь")
    plt.show()


def showF1gr(paths, metric="val_f1", text='F1-мера',
         legend=None):
    hists = []
    linestyle = ['-', '--', '-.', ':', '']
    colors = ['b', 'c', 'g', 'r', 'm', 'y', 'k', 'w']

    for path in paths:
        if os.path.exists(path):
            with open(path, "rb") as f:
                hists.append(pickle.load(f)[metric])

    if len(hists)==0:
        return

    maxepoch = max(len(x) for x in hists)
    epoch = range(1, maxepoch + 1)

    plt.figure(figsize=(11, 5))
    plt.xticks(epoch)

    # легенда
    if legend is None:
        legend = [str(x) for x in range(len(hists))]

    # задаем внешний вид линий
    for i, history in enumerate(hists):
        plt.plot(epoch, history, "{}{}".format(colors[(len(colors) - 1) % (2 * i + 1)],
                                               linestyle[(len(colors) - 1) % (2 * i + 1)]),
                 label=str(legend[i]))


    # подписи осей
    plt.xlabel('Эпохи')
    plt.ylabel(text)

    # подписи к графику
    for history in hists:
        plt.annotate('{:.4f}'.format(history[len(history) - 1]),
                     xy=(len(history), history[len(history) - 1]))

    plt.legend(loc='best')

    plt.title("График F1-меры")
    plt.show()

def summary(clasdata, w2vdata, cnndata):
    plt.figure(figsize=(5, 30))
    x = clasdata['Классификатор']
    y = clasdata['Ср F1']
    plt.subplot(3, 1, 1)
    plt.title("Классический подход")
    plt.barh(x, y)

    x = w2vdata['Классификатор']
    y = w2vdata['Ср F1']
    plt.subplot(3, 1, 2)
    plt.title("Word2vec подход")
    plt.barh(x, y)

    x = cnndata['Классификатор']
    y = cnndata['Ср F1']
    plt.subplot(3, 1, 3)
    plt.title("Сверточная нейронная сеть")
    plt.barh(x, y)

    plt.show()

