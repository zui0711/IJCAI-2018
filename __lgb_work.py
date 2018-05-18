import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm.sklearn import LGBMClassifier
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

def modelfit(alg, dtrain, predictors, cv_folds=5, early_stopping_rounds=50):

    print("CV start...")
    t1 = time.clock()
    lgb_param = alg.get_params()
    lgbtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain[column_name_label].values)
    cvresult = lgb.cv(lgb_param, lgbtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
        metrics='binary_logloss', early_stopping_rounds=early_stopping_rounds)
    print("\n\ntest-min: [%d] logloss %f"%(len(cvresult.items()[0][1]) - 1, cvresult.items()[0][1][-1]))

    alg.set_params(n_estimators=len(cvresult.items()[0][1]))

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

    feat_imp = pd.Series(alg.feature_importances_).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()


# predictors = list(set(column_name_x).union(set(column_name_id)))
predictors = column_name_x
print(len(predictors))

# xgb parameter tuning
# https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
#
# param_test1 = {
#     'feature_fraction': [0.8, 0.9],
#     'bagging_fraction': [0.8, 0.9]
# }
#
# gsearch1 = GridSearchCV(
#     estimator=LGBMClassifier(
#         n_estimators=78,
#         task='train',
#         boosting_type='gbdt',
#         objective='binary',
#         metric='binary_logloss',
#         learning_rate=0.1,
#         num_leaves=64,
#         feature_fraction=0.9,
#         bagging_fraction=0.9,
#         bagging_freq=5,
#         random_state=1337
#     ),
#     param_grid = param_test1,
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
    LGBMClassifier(
        task='train',
        boosting_type='gbdt',
        objective='binary',
        metric='binary_logloss',
        learning_rate=0.1,
        num_leaves=64,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        random_state=1337
    ),
    train,
    predictors)

"""
from data_split import data_split
from feature_extract import feat_extract

train1 = pd.read_csv('feats/train1.csv')
train2 = pd.read_csv('feats/train2.csv')
test1 = pd.read_csv('feats/test1.csv')
test2 = pd.read_csv('feats/test2.csv')

train_online = pd.read_csv('feats/train_online.csv')
test_online = pd.read_csv('feats/test_online.csv')

lgb_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'learning_rate': 0.1,
    'num_leaves': 64,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 5,
    'random_state': 1337
}


def lgb_cv():
    met = []
    for i, (train, test) in enumerate([(train1, test1), (train2, test2)]):
        trainx = train.drop(['instance_id', 'is_trade'], axis=1).values
        trainy = train['is_trade'].values
        testx = test.drop(['instance_id', 'is_trade'], axis=1).values
        testy = test['is_trade'].values

        dtrain = lgb.Dataset(trainx, label=trainy)
        dtest = lgb.Dataset(testx, label=testy)
        lgbresult = lgb.train(lgb_params, dtrain, num_boost_round=1000, valid_sets=[dtrain, dtest], early_stopping_rounds=50, verbose_eval=50)
        this_logloss = lgbresult.best_score['valid_1']['binary_logloss']
        print("Best iteration: [%d] %f" % (lgbresult.best_iteration, this_logloss))

        met.append(this_logloss)

    print '**************************************************'
    print met
    print 'average: %f' % np.mean(met)

def lgb_submit():
    trainx = train_online.drop(['instance_id', 'is_trade'], axis=1).values
    trainy = train_online['is_trade'].values
    testx = test_online.drop('instance_id', axis=1).values
    print trainx.shape, testx.shape

    dtrain = lgb.Dataset(trainx, label=trainy)
    dtest = lgb.Dataset(testx)
    lgbmodel = lgb.train(lgb_params, dtrain, num_boost_round=1000, verbose_eval=50)
    preprob = lgbmodel.predict(dtest)

    test_online["predicted_score"] = preprob
    test_online.to_csv("ans/" + 'XGB' + time.strftime(' %Y-%m-%d %H-%M-%S.csv', time.localtime(time.time())),
                       columns=["instance_id", "predicted_score"],
                       index=False)

# {a:[1,2], b:[1,2]}
# def lgb_gridsearchCV(tuning_params, params):
#
