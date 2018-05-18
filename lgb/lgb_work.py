
import pandas as pd
import numpy as np
import lightgbm as lgb
# from xgboost.sklearn import XGBClassifier
from sklearn import metrics   #Additional scklearn functions
from sklearn.model_selection import GridSearchCV   #Perforing grid search
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import time
from scipy import sparse
import matplotlib.pylab as plt

from preprocess_lgb import data_split

def offline_cv(dataset, lgb_params):
    met = []
    for i, (train, test) in enumerate(dataset):
        trainy = train['is_trade']
        trainx = train.drop(['instance_id', 'is_trade'], axis=1)
        testy = test['is_trade']
        testx = test.drop(['instance_id', 'is_trade'], axis=1)
        print trainx.shape, testx.shape

        enc = OneHotEncoder()
        for j, feat in enumerate(list(trainx.columns)):
            enc.fit(np.concatenate([train[feat], test[feat]]).reshape(-1, 1))
            x_train = enc.transform(train[feat].reshape(-1, 1))
            x_test = enc.transform(test[feat].reshape(-1, 1))

            if j == 0:
                X_train, X_test = x_train, x_test
            else:
                X_train, X_test = sparse.hstack((X_train, x_train)), sparse.hstack((X_test, x_test))

        trainx, testx = X_train, X_test
        print trainx.shape

        t1 = time.time()
        dtrain = lgb.Dataset(trainx, label=trainy)#, feature_name=trainx.columns)
        dtest = lgb.Dataset(testx, label=testy)#, feature_name=testx.columns)
        callback_dic = {}
        lgbresult = lgb.train(lgb_params, dtrain, num_boost_round=100, valid_sets=[dtest],
                              verbose_eval=50, callbacks=[lgb.callback.record_evaluation(callback_dic)])


        print 'cost time: %0.2f s' % (time.time() - t1)
        evals = callback_dic['valid_0']['binary_logloss']
        print("%d Best iteration: [%d] %f\n" % (i, np.array(evals).argmin(), np.array(evals).min()))
        met.append(evals)

        # lgb.plot_importance(lgbresult)
        # plt.show()

        # feat_imp = pd.Series(lgbresult.feature_importance()).sort_values(ascending=False)
        # feat_imp.plot(kind='bar', title='Feature Importances')
        # plt.show()

    print '**************************************************'
    cv_result = np.mean(met, axis=0)
    print "Best iteration: [%d] %f" % (cv_result.argmin(), cv_result.min())


def online_submit(train_online, test_online, lgb_params):
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

    dtrain = lgb.Dataset(trainx, label=trainy)
    dtest = testx
    lgbmodel = lgb.train(lgb_params, dtrain, num_boost_round=95)
    preprob = lgbmodel.predict(dtest)

    test_online["predicted_score"] = preprob
    test_online.to_csv("../ans/" + 'LGB' + time.strftime(' %Y-%m-%d %H-%M-%S.csv', time.localtime(time.time())),
                       columns=["instance_id", "predicted_score"],
                       index=False,
                       sep=" ")

if __name__ == '__main__':
    # data_split()

    GridSearch = False
    OFFLINE = False

    lgb_params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.1,
        'num_leaves': 32,
        'seed': 1337
    }

    if OFFLINE:
        dataset = []
        cv_ans = {}
        for i in range(0, 4):
            dataset.append((pd.read_csv('../xgb/train%d.csv' % i), pd.read_csv('../xgb/test%d.csv' % i)))
        if GridSearch:
            for max_depth in [3, 4, 5]:
                for min_child_weight in [1, 2, 3]:
                    print '\n\n******************************'
                    print 'max_depth: %d, min_child_weight: %d' % (max_depth, min_child_weight)
                    print '******************************'

                    # xgb_params['max_depth'] = max_depth
                    # xgb_params['min_child_weight'] = min_child_weight
                    # cv_ans[(max_depth, min_child_weight)] = offline_cv(dataset, xgb_params)

            for item in sorted(cv_ans.items(), key=lambda x: x[1], reverse=False):
                print item
        else:
            offline_cv(dataset, lgb_params)

    else:
        train_online = pd.read_csv('../xgb/train_online.csv')
        test_online = pd.read_csv('../xgb/test_online.csv')
        print train_online.columns
        online_submit(train_online, test_online, lgb_params)