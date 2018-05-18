# coding=utf-8

import pandas as pd
import numpy as np
import time
from sklearn import metrics, preprocessing
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import math
from scipy import sparse
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from lightgbm.sklearn import LGBMClassifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

from preprocess_lr import data_split




xgb_params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',  # 逻辑回归问题
    'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth': 5,  # 构建树的深度，越大越容易过拟合
    # 'lambda':2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample': 0.8,  # 随机采样训练样本
    'colsample_bytree': 0.8,  # 生成树时进行的列采样
    'min_child_weight': 1,
    'eta': 0.1,  # 如同学习率
    'seed': 1337,
    "silent": 1,
    'eval_metric': 'logloss',
    'n_job': -1,
    'tree_method': 'gpu_hist'
}

def offline_cv(dataset):
    met = []
    for i, (train, test) in enumerate(dataset):
        # for name in ['shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery',
        #              'shop_score_description']:
        #     train[name] = train[name].map(lambda x: math.pow(1000, x))
        #     test[name] = test[name].map(lambda x: math.pow(1000, x))

        trainy = train.pop('is_trade')
        trainx = train.drop('instance_id', axis=1)
        testy = test.pop('is_trade')
        testx = test.drop('instance_id', axis=1)
        print trainx.shape, testx.shape

        enc = OneHotEncoder()
        for i, feat in enumerate(list(trainx.columns)):
            enc.fit(np.concatenate([train[feat], test[feat]]).reshape(-1, 1))
            x_train = enc.transform(train[feat].reshape(-1, 1))
            x_test = enc.transform(test[feat].reshape(-1, 1))

            if i == 0:
                X_train, X_test = x_train, x_test
            else:
                X_train, X_test = sparse.hstack((X_train, x_train)), sparse.hstack((X_test, x_test))

        trainx, testx = X_train, X_test
        print trainx.shape

        # scaler = preprocessing.StandardScaler().fit(np.concatenate((trainx, testx)))
        # trainx = scaler.transform(trainx)
        # testx = scaler.transform(testx)

        lrmodel = LogisticRegression(C=0.08, random_state=1337)
        lrmodel.fit(trainx, trainy)
        preprob = lrmodel.predict_proba(testx)[:, 1]
        evals = metrics.log_loss(testy, preprob)

        # dtrain = lgb.Dataset(trainx, label=trainy)
        # dtest = lgb.Dataset(testx, label=testy)
        # lgbresult = lgb.train(lgb_params, dtrain, num_boost_round=1000, valid_sets=[dtrain, dtest], early_stopping_rounds=50, verbose_eval=50)
        # this_logloss = lgbresult.best_score['valid_1']['binary_logloss']
        # print("Best iteration: [%d] %f" % (lgbresult.best_iteration, this_logloss))

        # dtrain = xgb.DMatrix(trainx, label=trainy, feature_names=train.drop(['instance_id', 'is_trade'], axis=1).columns)
        # dtest = xgb.DMatrix(testx, label=testy, feature_names=test.drop(['instance_id', 'is_trade'], axis=1).columns)
        # callback_dic = {}
        # xgbresult = xgb.train(xgb_params, dtrain, num_boost_round=150, evals=[(dtrain, 'train'), (dtest, 'eval')],
        #                       verbose_eval=50, callbacks=[xgb.callback.record_evaluation(callback_dic)])
        # print callback_dic
        # this_logloss = xgbresult.best_score
        # print xgbresult.best_ntree_limit
        # evals = callback_dic['eval']['logloss']
        # print("Best iteration: [%d] %f" % (np.mean(evals), evals[xgbresult.best_iteration]))

        print evals
        met.append(evals)

    print '**************************************************'
    # print 'average: %f' % np.mean(np.array(met))
    cv_result = np.mean(met, axis=0)
    print met
    print "Best iteration: [%d] %0.8f" % (cv_result.argmin(), cv_result.min())

def online_submit(train_online, test_online):
    trainy = train_online.pop('is_trade')
    trainx = train_online.drop('instance_id', axis=1)
    testx = test_online.drop('instance_id', axis=1)
    print trainx.shape, testx.shape

    enc = OneHotEncoder()
    for i, feat in enumerate(list(trainx.columns)):
        enc.fit(np.concatenate([train_online[feat], test_online[feat]]).reshape(-1, 1))
        x_train = enc.transform(train_online[feat].reshape(-1, 1))
        x_test = enc.transform(test_online[feat].reshape(-1, 1))

        if i == 0:
            X_train, X_test = x_train, x_test
        else:
            X_train, X_test = sparse.hstack((X_train, x_train)), sparse.hstack((X_test, x_test))

    trainx, testx = X_train, X_test
    print trainx.shape

    # scaler = preprocessing.StandardScaler().fit(np.concatenate((trainx, testx)))
    # trainx = scaler.transform(trainx)
    # testx = scaler.transform(testx)

    lrmodel = LogisticRegression(C=0.08, random_state=1337)
    lrmodel.fit(trainx, trainy)
    preprob = lrmodel.predict_proba(testx)[:, 1]

    # dtrain = lgb.Dataset(trainx, label=trainy)
    # dtest = lgb.Dataset(testx)
    # lgbmodel = lgb.train(lgb_params, dtrain, num_boost_round=1000, verbose_eval=50)
    # preprob = lgbmodel.predict(dtest)

    # dtrain = xgb.DMatrix(trainx, label=trainy)
    # dtest = xgb.DMatrix(testx)
    # xgbmodel = xgb.train(xgb_params, dtrain, num_boost_round=100, verbose_eval=50)
    # preprob = xgbmodel.predict(dtest)

    test_online["predicted_score"] = preprob
    test_online.to_csv("../ans/" + 'LR' + time.strftime(' %Y-%m-%d %H-%M-%S.csv', time.localtime(time.time())),
                       columns=["instance_id", "predicted_score"],
                       index=False,
                       sep=" ")

if __name__ == '__main__':
    # data_split()

    OFFLINE = True

    if OFFLINE:
        dataset = []
        for i in range(0, 4):
            dataset.append((pd.read_csv('train%d.csv'%i), pd.read_csv('test%d.csv' % i)))

        offline_cv(dataset)
    else:
        train_online = pd.read_csv('train_online.csv')
        test_online = pd.read_csv('test_online.csv')
        print train_online.columns

        # for name in ['shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery',
        #              'shop_score_description']:
        #     train_online[name] = train_online[name].map(lambda x: math.pow(1000, x))
        #     test_online[name] = test_online[name].map(lambda x: math.pow(1000, x))

        online_submit(train_online, test_online)