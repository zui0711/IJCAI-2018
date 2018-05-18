# coding=utf-8
import pandas as pd
import numpy as np
import xgboost as xgb
# from xgboost.sklearn import XGBClassifier
from sklearn import metrics   #Additional scklearn functions
from sklearn.model_selection import GridSearchCV   #Perforing grid search
import time

import matplotlib.pylab as plt
"""
train = pd.read_csv('data/round1_ijcai_18_train_20180301.txt', sep=" ")

column_name_id = ['instance_id', 'item_id', 'item_brand_id', 'item_city_id', 'user_id', 'user_gender_id', 'user_occupation_id', 'context_id', 'context_page_id', 'shop_id']
column_name_x = ['item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level', 'user_age_level', 'user_star_level', 'context_timestamp', 'shop_review_num_level', 'shop_review_positive_rate', 'shop_star_level', 'shop_score_service', 'shop_score_delivery', 'shop_score_description']

column_name_p = ['item_category_list', 'item_property_list', 'predict_category_property']

column_name_label = 'is_trade'

def modelfit(alg, dtrain, predictors, cv_folds=4, early_stopping_rounds=50):
    print("CV start...")
    t1 = time.clock()
    xgb_param = alg.get_params()
    # xgb_param = alg.get_xgb_params()
    xgbtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[column_name_label].values)
    cvresult = xgb.cv(xgb_param, xgbtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
        metrics='logloss', early_stopping_rounds=early_stopping_rounds)
    print("test-min: [%d] logloss %f"%(cvresult.values.argmin(0)[0], cvresult.values[cvresult.values.argmin(0)[0]][0]))

    alg.set_params(n_estimators=cvresult.shape[0])

    t2 = time.clock()
    print(t2 - t1)

    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[column_name_label],eval_metric='logloss')

    print(time.clock() - t2)

    #Predict training set:
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]

    #Print model report:
    print("\nModel Report")
    # print("Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions))
    print("Logloss (Train): %f" % metrics.log_loss(dtrain[column_name_label], dtrain_predprob))

    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()


# predictors = list(set(column_name_x).union(set(column_name_id)))
predictors = column_name_x
print(len(predictors))

# xgb parameter tuning
# https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

# param_test1 = {
#     'max_depth': range(4, 6, 1),
#     'min_child_weight': range(1, 5, 2)
# }
#
# gsearch1 = GridSearchCV(
#     estimator=XGBClassifier(
#         learning_rate=0.1,
#         n_estimators=225,
#         max_depth=5,
#         min_child_weight=1,
#         gamma=0,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         objective='binary:logistic',
#         n_jobs=4,
#         scale_pos_weight=1,
#         random_state=1337,
#         scoring='logloss'
#     ),
#     param_grid=param_test1,
#     scoring='neg_log_loss',
#     n_jobs=-1,
#     iid=False,
#     cv=5)
#
# gsearch1.fit(train[predictors],train[column_name_label])
#
# print(gsearch1.grid_scores_)
# print(gsearch1.best_params_)
# print(gsearch1.best_score_)

# cv_result = pd.DataFrame.from_dict(gsearch1.cv_results_)
# print(cv_result)

modelfit(
    XGBClassifier(
        learning_rate=0.1,
        n_estimators=1000,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=-1,
        # n_jobs=-1,
        scale_pos_weight=1,
        seed=1337,
        # random_state=1337.
        silent=1
    ),
    train,
    predictors)

"""

from preprocess_xgb import data_split

def offline_cv(dataset, xgb_params):
    met = []
    for i, (train, test) in enumerate(dataset):
        trainy = train['is_trade']
        trainx = train.drop(['instance_id', 'is_trade'], axis=1)
        testy = test['is_trade']
        testx = test.drop(['instance_id', 'is_trade'], axis=1)
        print trainx.shape, testx.shape

        t1 = time.time()
        dtrain = xgb.DMatrix(trainx, label=trainy)#, feature_names=trainx.columns)
        dtest = xgb.DMatrix(testx, label=testy)#, feature_names=testx.columns)
        callback_dic = {}
        xgbresult = xgb.train(xgb_params, dtrain, num_boost_round=300, evals=[(dtrain, 'train'), (dtest, 'eval')],
                              verbose_eval=50, callbacks=[xgb.callback.record_evaluation(callback_dic)])

        print 'cost time: %0.2f s'%(time.time() - t1)
        evals = callback_dic['eval']['logloss']
        print("%d Best iteration: [%d] %0.8f\n" % (i, np.array(evals).argmin(), np.array(evals).min()))
        met.append(evals)

        # xgb.plot_importance(xgbresult)
        # plt.show()

        # for item in sorted(xgbresult.get_fscore().items(), key=lambda x: x[1], reverse=False):
        #     print item
        #
        # feat_imp = pd.Series(xgbresult.get_fscore()).sort_values(ascending=False)
        # feat_imp.plot(kind='bar', title='Feature Importances')
        # plt.show()

    print '**************************************************'
    cv_result = np.mean(met, axis=0)
    print "Best iteration: [%d] %0.8f" % (cv_result.argmin(), cv_result.min())
    return cv_result.min()


def online_submit(train_online, test_online, xgb_params):
    trainy = train_online.pop('is_trade')
    trainx = train_online.drop('instance_id', axis=1)
    testx = test_online.drop('instance_id', axis=1)

    print trainx.shape, testx.shape

    dtrain = xgb.DMatrix(trainx, label=trainy)
    dtest = xgb.DMatrix(testx)
    xgbmodel = xgb.train(xgb_params, dtrain, num_boost_round=283)
    preprob = xgbmodel.predict(dtest)

    test_online["predicted_score"] = preprob
    test_online.to_csv("../ans/" + 'XGB' + time.strftime(' %Y-%m-%d %H-%M-%S.csv', time.localtime(time.time())),
                       columns=["instance_id", "predicted_score"],
                       index=False,
                       sep=" ")

if __name__ == '__main__':
    # data_split()

    OFFLINE = False
    GridSearch = False

    xgb_params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',  # 逻辑回归问题
        'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
        'max_depth': 4,  # 构建树的深度，越大越容易过拟合
        # 'lambda':2, # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        'subsample': 0.8,  # 随机采样训练样本
        'colsample_bytree': 0.8,  # 生成树时进行的列采样
        'min_child_weight': 1,
        'eta': 0.05,  # 如同学习率
        'seed': 1337,
        "silent": 1,
        'eval_metric': 'logloss',
        'n_job': -1,
        'tree_method': 'gpu_hist'
    }

    if OFFLINE:
        dataset = []
        cv_ans = {}
        for i in range(0, 4):
            dataset.append((pd.read_csv('train%d.csv' % i), pd.read_csv('test%d.csv' % i)))
        if GridSearch:
            for max_depth in [2,3,4,5,6,7]:
                for min_child_weight in [1,2,3,4,5]:
                    print '\n\n******************************'
                    print 'max_depth: %d, min_child_weight: %d'%(max_depth, min_child_weight)
                    print '******************************'

                    xgb_params['max_depth'] = max_depth
                    xgb_params['min_child_weight'] = min_child_weight
                    cv_ans[(max_depth, min_child_weight)] = offline_cv(dataset, xgb_params)

            for item in sorted(cv_ans.items(), key=lambda x: x[1], reverse=False):
                print item
        else:
            offline_cv(dataset, xgb_params)

    else:
        train_online = pd.read_csv('train_online.csv')
        test_online = pd.read_csv('test_online.csv')
        print train_online.columns
        online_submit(train_online, test_online, xgb_params)