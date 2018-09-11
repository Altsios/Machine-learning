# -*- coding: utf-8 -*-
import json
import warnings
from sklearn import metrics, model_selection as ms
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import pickle
import psycopg2 as ps

warnings.simplefilter("ignore")


Classifiers = {"GaussianNB":GaussianNB,
                "AdaBoostClassifier": AdaBoostClassifier,
                "DecisionTreeClassifier": DecisionTreeClassifier,
                "SVM": SVC,
                "LogisticRegressionCV": LogisticRegressionCV,
                "RandomForestClassifier": RandomForestClassifier}

def MakeParams(classif_name):

    params = []

    if classif_name == "GaussianNB":
            params.append({"priors": None})


    elif classif_name == "LogisticRegressionCV":
            params.append({"solver": 'saga',
                           "random_state": 13,
                           'fit_intercept': True,
                           'n_jobs': -1,
                           "refit": True})


    elif classif_name == "RandomForestClassifier":

        for c, e, m in [(c, e, m)
                        for c in ['gini', 'entropy']
                        for e in [30, 65, 100]
                        for m in ['sqrt', 'log2']]:
            params.append({"criterion": c,
                           "n_estimators": e,
                           "max_features": m,
                           "random_state": 13,
                           "min_samples_split": 2,
                           "n_jobs": -1})


    elif classif_name == "SVM":
        for k in ["rbf", "linear", "sigmoid"]:
            params.append({"kernel": k,
                           "random_state": 13})


    elif classif_name == "AdaBoostClassifier":

        for e in [30, 65, 100]:
            params.append({"n_estimators": e,
                           "algorithm": "SAMME.R",
                           "random_state": 13})

    else:# "DecisionTreeClassifier"
        for c, s, ml, ms in [(c, s, ml, ms)
                             for c in ['gini', 'entropy']
                             for s in ['random', 'best']
                             for ml in [1,5]
                             for ms in [2,8]]:
            params.append({"criterion": c,
                           "splitter": s,
                           "min_samples_leaf": ml,
                           "min_samples_split": ms,
                            "random_state": 13})




    return params


def OneCalc(par, name,ngram, sentiments, classif_name,save,conn,tab,bestf1,message_callback):

    model = Classifiers[classif_name](**par)
    message_callback.emit("  Параметры: {}".format(par))
    predictions = ms.cross_val_predict(model, ngram.vectors, sentiments, cv=5, n_jobs=-1)

    newf1=metrics.f1_score(sentiments, predictions, average='weighted')
    prec=metrics.precision_score(sentiments, predictions, average='weighted')
    recall=metrics.recall_score(sentiments, predictions, average='weighted')

    if save and newf1>bestf1:
        with open('./Classifiers/classifier{}_{}.pkl'.format(classif_name,tab), 'wb') as fid:
            pickle.dump(model, fid)

    message_callback.emit("insert into {} values ('{}','{}','{}','{}','{}',{},{},{})".
                         format(tab, name, ngram.type, ngram.metrics,
                                classif_name, json.dumps(par), prec, recall, newf1))
    if newf1 > bestf1:
        bestf1=newf1


        with conn.cursor() as cur:
            cur.execute("insert into {} values ('{}','{}','{}','{}','{}',{},{},{})".
                    format(tab, name, ngram.type, ngram.metrics,
                            classif_name, json.dumps(par), prec, recall, newf1))
    return bestf1



def CalcMet(message_callback,name, ngram, sentiments, tab,
            cls, save,conn_str):

    bestf1=-1
    for classif_name in cls.keys():

        if classif_name in Classifiers.keys():

            message_callback.emit("Классификатор: {}".format(classif_name))

            if cls[classif_name]=="Все":
                params = MakeParams(classif_name)
            else:
                params=cls[classif_name].split(', ')
            try:
                for par in params:
                    bestf1=OneCalc(par, name, ngram, sentiments,
                        classif_name, save, conn_str, tab,bestf1,message_callback)
            except Exception:
                message_callback.emit("Неверные параметры ли малое число элементов в выборке")
