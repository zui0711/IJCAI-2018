# coding=utf-8

import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder

raw_feats = [
    'instance_id',
    'item_id',
    'item_category_list',
    'item_property_list',
    'item_brand_id',
    'item_city_id',
    'item_price_level',
    'item_sales_level',
    'item_collected_level',
    'item_pv_level',
    'user_id',
    'user_gender_id',
    'user_age_level',
    'user_occupation_id',
    'user_star_level',
    'context_id',
    'context_timestamp',
    'context_page_id',
    'predict_category_property',
    'shop_id',
    'shop_review_num_level',
    'shop_review_positive_rate',
    'shop_star_level',
    'shop_score_service',
    'shop_score_delivery',
    'shop_score_description',
    'is_trade'
]

feats_without_prop = [
'instance_id',
'item_id',
# 'item_category_list',
# 'item_property_list',
'item_brand_id',
'item_city_id',
'item_price_level',
'item_sales_level',
'item_collected_level',
'item_pv_level',
'user_id',
'user_gender_id',
'user_age_level',
'user_occupation_id',
'user_star_level',
'context_id',
'context_timestamp',
'context_page_id',
# 'predict_category_property',
'shop_id',
'shop_review_num_level',
'shop_review_positive_rate',
'shop_star_level',
'shop_score_service',
'shop_score_delivery',
'shop_score_description',
'is_trade'
]

feats1 = [
'item_price_level',
'item_sales_level',
'item_collected_level',
'item_pv_level',
'user_age_level',
'user_star_level',
'context_timestamp',
'shop_review_num_level',
'shop_review_positive_rate',
'shop_star_level',
'shop_score_service',
'shop_score_delivery',
'shop_score_description',
'is_trade'
]


def add_date(df):
    time_df = pd.DataFrame(columns=["day", "hour"])
    time_df["day"] = df["context_timestamp"].map(lambda x: time.localtime(x).tm_mday)
    time_df["hour"] = df["context_timestamp"].map(lambda x: time.localtime(x).tm_hour)
    df = pd.concat([df, time_df], axis=1)
    return df


def get_prop(df):
    print('item_category_list_ing')
    for i in range(3):
        df['category_%d' % (i)] = df['item_category_list'].apply(
            lambda x: x.split(";")[i] if len(x.split(";")) > i else "0"
        )
    # del df['item_category_list']

    print('item_property_list_ing')
    for i in range(3):
        df['property_%d' % (i)] = df['item_property_list'].apply(
            lambda x: x.split(";")[i] if len(x.split(";")) > i else "0"
        )
    # del df['item_property_list']

    print('predict_category_property_ing')
    for i in range(3):
        df['predict_category_%d' % (i)] = df['predict_category_property'].apply(
            lambda x: str(x.split(";")[i]).split(":")[0] if len(x.split(";")) > i else "0"
        )

    return df


def feat_extract():
    print '********************************************'
    print 'feature extract'
    print '********************************************\n'


    column_name_x = ['item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level', 'user_age_level', 'user_star_level', 'context_timestamp', 'shop_review_num_level', 'shop_review_positive_rate', 'shop_star_level', 'shop_score_service', 'shop_score_delivery', 'shop_score_description']
    column_name_label = 'is_trade'

    column_name_id = ['instance_id', 'item_id', 'item_brand_id', 'item_city_id', 'user_id', 'user_gender_id', 'user_occupation_id', 'context_id', 'context_page_id', 'shop_id']
    column_name_p = ['item_category_list', 'item_property_list', 'predict_category_property']


    # dtrain = get_prop(dtrain)
    # dtest = get_prop(dtest)

    # dtrain.filla()

    # for n in dtrain.columns:
    #     dtrain[n].replace(-1, np.nan, inplace=True)
    #
    # for n in dtest.columns:
    #     dtrain[n].replace(-1, np.nan, inplace=True)

    feats = feats1 + ['instance_id', 'day', 'hour']
    # feats = feats1 +['instance_id', 'day', 'hour',
    #                  'category_0', 'category_1', 'category_2',
    #                  'property_0', 'property_1', 'property_2',
    #                  'predict_category_0', 'predict_category_1', 'predict_category_2']
    # feats = feats_without_prop + ['day', 'hour',
    #                               'category_0', 'category_1', 'category_2',
    #                               'property_0', 'property_1', 'property_2',
    #                               'predict_category_0', 'predict_category_1', 'predict_category_2']

    # dtrain.to_csv("feats/dtrain.csv", columns=feats, index=False)
    # feats.remove('is_trade')
    # dtest.to_csv("feats/dtest.csv", columns=feats, index=False)


def data_split():
    print '********************************************'
    print 'data split'
    print '********************************************\n'

    dtrain = pd.read_csv('data/round1_ijcai_18_train_20180301.txt', sep=" ")
    dtest = pd.read_csv("data/round1_ijcai_18_test_a_20180301.txt", sep=" ")

    for df in [dtrain, dtest]:
        add_date(df)
        get_prop(df)

    # 18 19 20 21 22 23 || 24
    train1 = dtrain[(dtrain['day'] >= 18) & (dtrain['day'] <= 21)]
    train1['day'] = train1['day'].map(lambda x: x-18)
    test1 = dtrain[dtrain['day'] == 22]
    test1['day'] = test1['day'].map(lambda x: x-18)

    train2 = dtrain[(dtrain['day'] >= 19) & (dtrain['day'] <= 22)]
    train2['day'] = train2['day'].map(lambda x: x-19)
    test2 = dtrain[dtrain['day'] == 23]
    test2['day'] = test2['day'].map(lambda x: x-19)

    train_online = dtrain[(dtrain['day'] >= 20) & (dtrain['day'] <= 23)]
    train_online['day'] = train_online['day'].map(lambda x: x-20)
    test_online = dtest
    test_online['day'] = test_online['day'].map(lambda x: x-20)

    for (train, test) in [(train1, test1), (train2, test2), (train_online, test_online)]:
        le = LabelEncoder()
        for name in ['category_0', 'category_1', 'category_2',
                     'property_0', 'property_1', 'property_2',
                     'predict_category_0', 'predict_category_1', 'predict_category_2']:
            le.fit_transform((list(train[name]) + list(test[name])))
            train[name] = train[name].map(lambda x:le.transform(train[name]))
            test[name] = test[name].map(lambda x:le.transform(test[name]))


    train1.to_csv('feats/train1.csv', columns=train1.columns, index=False)
    train2.to_csv('feats/train2.csv', columns=train2.columns, index=False)
    test1.to_csv('feats/test1.csv', columns=test1.columns, index=False)
    test2.to_csv('feats/test2.csv', columns=test2.columns, index=False)

    train_online.to_csv('feats/train_online.csv', columns=train_online.columns, index=False)
    test_online.to_csv('feats/test_online.csv', columns=test_online.columns, index=False)

if __name__ == '__main__':
    feat_extract()