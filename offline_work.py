# coding=utf-8

import numpy as np
np.random.seed(1337)

import pandas as pd

df = pd.read_csv('data/round1_ijcai_18_train_20180301.txt', sep=" ")

column_name_x = ['item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level', 'user_age_level', 'user_star_level', 'context_timestamp', 'shop_review_num_level', 'shop_review_positive_rate', 'shop_star_level', 'shop_score_service', 'shop_score_delivery', 'shop_score_description']
column_name_label = 'is_trade'

column_name_id = ['instance_id', 'item_id', 'item_brand_id', 'item_city_id', 'user_id', 'user_gender_id', 'user_occupation_id', 'context_id', 'context_page_id', 'shop_id']
column_name_p = ['item_category_list', 'item_property_list', 'predict_category_property']

y = df[column_name_label].values
df.drop(column_name_label, axis=1, inplace=True)
x = df[column_name_x].values
# x = df[list(set(column_name_x).union(set(column_name_id)))].values

# print(sum(y), len(y))

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import xgboost as xgb
import lightgbm as lgb

from sklearn import preprocessing, metrics

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold, KFold


import matplotlib.pyplot as plot

# def mscale(train_x, train_y, test_x):
#     scalerX = preprocessing.MinMaxScaler(feature_range=(0, 1))
#     scalerY = preprocessing.MinMaxScaler(feature_range=(0, 1))
#     train_x = scalerX.fit_transform(train_x)
#     train_y = scalerY.fit_transform(train_y)
#     test_x = scalerX.transform(test_x)
#     return train_x, train_y, test_x, scalerX, scalerY

scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
x = scaler.fit_transform(x)

# from imblearn.combine import SMOTEENN
# from imblearn.over_sampling import SMOTE, RandomOverSampler
# sm = SMOTEENN()
# sm = SMOTE(ratio={1: sum(trainy)*2, 0: len(trainy)-sum(trainy)}, random_state=1337)
# trainx, trainy = sm.fit_sample(trainx, trainy)
# print(sum(trainy), len(trainy))

# from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1337)
# kf = KFold(n_splits=5, shuffle=True, random_state=1337)

for i, (train_index, test_index) in enumerate(skf.split(x, y)):
    if i == 0:
        trainx = x[train_index]
        trainy = y[train_index]
        testx = x[test_index]
        testy = y[test_index]

        # trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.2, random_state=1337)
        print(sum(trainy), len(trainy))
        print(sum(testy), len(testy))

        """
        LR
        """
        # logreg = LogisticRegressionCV(Cs=[1e5], cv=5, random_state=1337)
        # logreg.fit(trainx, trainy)
        # preprob = logreg.predict_proba(testx)[:,1]
        # print metrics.log_loss(testy, preprob)

        """
        KNN
        """
        model = Sequential()
        model.add(Dense(64, input_dim=13, activation='relu'))
        model.add(Dense(64, input_dim=13, activation='relu'))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(loss='binary_crossentropy', optimizer=optimizers.adam(lr=0.001), metrics=['binary_crossentropy'])
        model.fit(trainx, trainy, batch_size=256, epochs=20, validation_data=(testx, testy))

        """
        XGBOOST
        """
        # xgb_params = {
        #     'booster': 'gbtree',
        #     'objective': 'binary:logistic',  # 逻辑回归问题
        #     'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
        #     'max_depth': 5,  # 构建树的深度，越大越容易过拟合
        #     # 'lambda':2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        #     'subsample': 0.8,  # 随机采样训练样本
        #     'colsample_bytree': 0.8,  # 生成树时进行的列采样
        #     'min_child_weight': 1,
        #     'eta': 0.1,  # 如同学习率
        #     'seed': 1337,
        #     "silent": 1,
        #     'eval_metric': 'logloss',
        #     'n_job': -1
        # }
        # dtrain = xgb.DMatrix(trainx, label=trainy)
        # dtest = xgb.DMatrix(testx, label=testy)
        # model = xgb.train(xgb_params, dtrain, num_boost_round=5000, evals=[(dtrain, 'train'), (dtest, 'eval')], early_stopping_rounds=50)
        # xgb.plot_importance(model)
        # plot.show()

        """
        LIGHTGBM
        """
        # lgb_params = {
        #     'task': 'train',
        #     'boosting_type': 'gbdt',
        #     'objective': 'binary',
        #     'metric': 'binary_logloss',
        #     'learning_rate': 0.1,
        #     'num_leaves': 64,
        #     'feature_fraction': 0.9,
        #     'bagging_fraction': 0.9,
        #     'bagging_freq': 5,
        #     'random_state': 1337
        # }
        # dtrain = lgb.Dataset(trainx, label=trainy)
        # dtest = lgb.Dataset(testx, label=testy)
        # model = lgb.train(lgb_params, dtrain, num_boost_round=2000, valid_sets=[dtrain, dtest], early_stopping_rounds=50)
        # lgb.plot_importance(model)
        # plot.show()



# ensemble
# https://dnc1994.com/2016/04/rank-10-percent-in-first-kaggle-competition/