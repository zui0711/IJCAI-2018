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


feats = feats1 + ['instance_id', 'day', 'hour']

def add_date(df):
    print '********************************************'
    print 'add date...'
    print '********************************************\n'
    time_df = pd.DataFrame(columns=["day", "hour"])
    time_df["day"] = df["context_timestamp"].map(lambda x: time.localtime(x).tm_mday)
    time_df["hour"] = df["context_timestamp"].map(lambda x: time.localtime(x).tm_hour)
    df = pd.concat([df, time_df], axis=1)
    return df


def get_prop(df):
    print '********************************************'
    print 'get category, property...'
    print '********************************************\n'
    for i in range(1, 3):
        df['category_%d' % (i)] = df['item_category_list'].apply(
            lambda x: x.split(";")[i] if len(x.split(";")) > i else "0"
        )
    del df['item_category_list']
    # del df['category_0']

    for i in range(3):
        df['property_%d' % (i)] = df['item_property_list'].apply(
            lambda x: x.split(";")[i] if len(x.split(";")) > i else "0"
        )
    del df['item_property_list']

    for i in range(3):
        df['predict_category_%d' % (i)] = df['predict_category_property'].apply(
            lambda x: str(x.split(";")[i]).split(":")[0] if len(x.split(";")) > i else "0"
        )
    del df['predict_category_property']

    return df


def add_history(dtrain, dtest):
    for name in ['hour']:#, 'predict_category_0', 'category_1']:
        trade = dtrain.groupby([name, 'is_trade']).size().reset_index(name=name+'_trade')
        trade = trade[trade['is_trade'] == 1].drop('is_trade', axis=1)
        click = dtrain.groupby([name]).size().reset_index(name=name+'_click')
        all = pd.merge(trade, click, 'left', on=[name])
        all[name+'_trade_rate'] = all[name+'_trade'].astype(float) / all[name+'_click'].astype(float)
        del all[name+'_trade'], all[name+'_click']
        dtrain = pd.merge(dtrain, all, 'left', on=name)
        dtest = pd.merge(dtest, all, 'left', on=name)

    # hour
    # hour_trade = dtrain.groupby(['hour', 'is_trade']).size().reset_index(name='hour_trade')
    # hour_trade = hour_trade[hour_trade['is_trade'] == 1].drop('is_trade', axis=1)
    # hour_click = dtrain.groupby(['hour']).size().reset_index(name='hour_click')
    # hour = pd.merge(hour_trade, hour_click, 'left', on=['hour'])
    # hour = hour.astype(float)
    # hour['hour_trade_rate'] = hour['hour_trade'] / hour['hour_click']
    # # del hour['hour_click']
    # dtrain = pd.merge(dtrain, hour, 'left', on=['hour'])
    # dtest = pd.merge(dtest, hour, 'left', on=['hour'])
    # del hour, hour_click, hour_trade

    # item
    # item_id_trade = dtrain.groupby(['item_id', 'is_trade']).size().reset_index(name='item_id_trade')
    # item_id_trade = item_id_trade[item_id_trade['is_trade'] == 1].drop('is_trade', axis=1)
    # item_id_click = dtrain.groupby(['item_id']).size().reset_index(name='item_id_click')
    # item_id = pd.merge(item_id_trade, item_id_click, 'left', on=['item_id'])
    # item_id = item_id.astype(float)
    # item_id['item_id_trade_rate'] = item_id['item_id_trade'] / item_id['item_id_click']
    # # del item_id['item_id_click'], item_id['item_id_trade']
    # dtrain = pd.merge(dtrain, item_id, 'left', on=['item_id'])
    # dtest = pd.merge(dtest, item_id, 'left', on=['item_id'])
    # del item_id, item_id_click, item_id_trade

    # item_sales_trade = dtrain.groupby(['item_sales_level', 'is_trade']).size().reset_index(name='item_sales_trade')
    # item_sales_trade = item_sales_trade[item_sales_trade['is_trade'] == 1].drop('is_trade', axis=1)
    # item_sales_click = dtrain.groupby(['item_sales_level']).size().reset_index(name='item_sales_click')
    # item_sales = pd.merge(item_sales_trade, item_sales_click, 'left', on=['item_sales_level'])
    # item_sales = item_sales.astype(float)
    # item_sales['item_sales_trade_rate'] = item_sales['item_sales_trade'] / item_sales['item_sales_click']
    # del item_sales['item_sales_click'], item_sales['item_sales_trade']
    # dtrain = pd.merge(dtrain, item_sales, 'left', on=['item_sales_level'])
    # dtest = pd.merge(dtest, item_sales, 'left', on=['item_sales_level'])
    # del item_sales, item_sales_click, item_sales_trade

    # item_brand_trade = dtrain.groupby(['item_brand_id', 'is_trade']).size().reset_index(name='item_brand_trade')
    # item_brand_trade = item_brand_trade[item_brand_trade['is_trade'] == 1].drop('is_trade', axis=1)
    # item_brand_click = dtrain.groupby(['item_brand_id']).size().reset_index(name='item_brand_click')
    # item_brand = pd.merge(item_brand_trade, item_brand_click, 'left', on=['item_brand_id'])
    # item_brand = item_brand.astype(float)
    # item_brand['item_brand_trade_rate'] = item_brand['item_brand_trade'] / item_brand['item_brand_click']
    # dtrain = pd.merge(dtrain, item_brand, 'left', on=['item_brand_id'])
    # dtest = pd.merge(dtest, item_brand, 'left', on=['item_brand_id'])
    # del item_brand, item_brand_click, item_brand_trade

    # user
    # user_star_trade = dtrain.groupby(['user_star_level', 'is_trade']).size().reset_index(name='user_star_trade')
    # user_star_trade = user_star_trade[user_star_trade['is_trade'] == 1].drop('is_trade', axis=1)
    # user_star_click = dtrain.groupby(['user_star_level']).size().reset_index(name='user_star_click')
    # user_star = pd.merge(user_star_trade, user_star_click, 'left', on=['user_star_level'])
    # user_star = user_star.astype(float)
    # user_star['user_star_trade_rate'] = user_star['user_star_trade'] / user_star['user_star_click']
    # del user_star['user_star_click']
    # dtrain = pd.merge(dtrain, user_star, 'left', on=['user_star_level'])
    # dtest = pd.merge(dtest, user_star, 'left', on=['user_star_level'])
    # del user_star, user_star_click, user_star_trade
    #
    # user_age_trade = dtrain.groupby(['user_age_level', 'is_trade']).size().reset_index(name='user_age_trade')
    # user_age_trade = user_age_trade[user_age_trade['is_trade'] == 1].drop('is_trade', axis=1)
    # user_age_click = dtrain.groupby(['user_age_level']).size().reset_index(name='user_age_click')
    # user_age = pd.merge(user_age_trade, user_age_click, 'left', on=['user_age_level'])
    # user_age = user_age.astype(float)
    # user_age['user_age_trade_rate'] = user_age['user_age_trade'] / user_age['user_age_click']
    # del user_age['user_age_trade'], user_age['user_age_click']
    # dtrain = pd.merge(dtrain, user_age, 'left', on=['user_age_level'])
    # dtest = pd.merge(dtest, user_age, 'left', on=['user_age_level'])
    # del user_age, user_age_click, user_age_trade

    # shop
    # shop_delivery_trade = dtrain.groupby(['shop_score_delivery', 'is_trade']).size().reset_index(
    #     name='shop_delivery_trade')
    # shop_delivery_trade = shop_delivery_trade[shop_delivery_trade['is_trade'] == 1].drop('is_trade', axis=1)
    # shop_delivery_click = dtrain.groupby(['shop_score_delivery']).size().reset_index(name='shop_delivery_click')
    # shop_delivery = pd.merge(shop_delivery_trade, shop_delivery_click, 'left', on=['shop_score_delivery'])
    # shop_delivery = shop_delivery.astype(float)
    # shop_delivery['shop_delivery_trade_rate'] = shop_delivery['shop_delivery_trade'] / shop_delivery[
    #     'shop_delivery_click']
    # del shop_delivery['shop_delivery_trade'],shop_delivery['shop_delivery_click']
    # dtrain = pd.merge(dtrain, shop_delivery, 'left', on=['shop_score_delivery'])
    # dtest = pd.merge(dtest, shop_delivery, 'left', on=['shop_score_delivery'])
    # del shop_delivery, shop_delivery_click, shop_delivery_trade

    # shop_delivery_trade = dtrain.groupby(['shop_delivery_bin', 'is_trade']).size().reset_index(
    #     name='shop_delivery_trade')
    # shop_delivery_trade = shop_delivery_trade[shop_delivery_trade['is_trade'] == 1].drop('is_trade', axis=1)
    # shop_delivery_click = dtrain.groupby(['shop_delivery_bin']).size().reset_index(name='shop_delivery_click')
    # shop_delivery = pd.merge(shop_delivery_trade, shop_delivery_click, 'left', on=['shop_delivery_bin'])
    # # shop_delivery = shop_delivery.astype(float)
    # shop_delivery['shop_delivery_bin'] = shop_delivery['shop_delivery_bin'].astype(float)
    # shop_delivery['shop_delivery_trade_rate'] = shop_delivery['shop_delivery_trade'].astype(float) / shop_delivery[
    #     'shop_delivery_click'].astype(float)
    # del shop_delivery['shop_delivery_trade'], shop_delivery['shop_delivery_click']
    # dtrain = pd.merge(dtrain, shop_delivery, 'left', on=['shop_delivery_bin'])
    # dtest = pd.merge(dtest, shop_delivery, 'left', on=['shop_delivery_bin'])
    # del shop_delivery, shop_delivery_click, shop_delivery_trade

    # shop_description_trade = dtrain.groupby(['shop_score_description', 'is_trade']).size().reset_index(
    #     name='shop_description_trade')
    # shop_description_trade = shop_description_trade[shop_description_trade['is_trade'] == 1].drop('is_trade', axis=1)
    # shop_description_click = dtrain.groupby(['shop_score_description']).size().reset_index(
    #     name='shop_description_click')
    # shop_description = pd.merge(shop_description_trade, shop_description_click, 'left', on=['shop_score_description'])
    # shop_description = shop_description.astype(float)
    # shop_description['shop_description_trade_rate'] = shop_description['shop_description_trade'] / shop_description[
    #     'shop_description_click']
    # del shop_description['shop_description_trade']
    # dtrain = pd.merge(dtrain, shop_description, 'left', on=['shop_score_description'])
    # dtest = pd.merge(dtest, shop_description, 'left', on=['shop_score_description'])
    # del shop_description, shop_description_click, shop_description_trade
    # shop_description_trade = dtrain.groupby(['shop_description_bin', 'is_trade']).size().reset_index(
    #     name='shop_description_trade')
    # shop_description_trade = shop_description_trade[shop_description_trade['is_trade'] == 1].drop('is_trade', axis=1)
    # shop_description_click = dtrain.groupby(['shop_description_bin']).size().reset_index(name='shop_description_click')
    # shop_description = pd.merge(shop_description_trade, shop_description_click, 'left', on=['shop_description_bin'])
    # shop_description['shop_description_bin'] = shop_description['shop_description_bin'].astype(float)
    # shop_description['shop_description_trade_rate'] = shop_description['shop_description_trade'] / shop_description[
    #     'shop_description_click']
    # del shop_description['shop_description_trade'], shop_description['shop_description_click']
    # dtrain = pd.merge(dtrain, shop_description, 'left', on=['shop_description_bin'])
    # dtest = pd.merge(dtest, shop_description, 'left', on=['shop_description_bin'])
    # del shop_description, shop_description_click, shop_description_trade

    # shop_service_trade = dtrain.groupby(['shop_score_service', 'is_trade']).size().reset_index(
    #     name='shop_service_trade')
    # shop_service_trade = shop_service_trade[shop_service_trade['is_trade'] == 1].drop('is_trade', axis=1)
    # shop_service_click = dtrain.groupby(['shop_score_service']).size().reset_index(name='shop_service_click')
    # shop_service = pd.merge(shop_service_trade, shop_service_click, 'left', on=['shop_score_service'])
    # shop_service = shop_service.astype(float)
    # shop_service['shop_service_trade_rate'] = shop_service['shop_service_trade'] / shop_service[
    #     'shop_service_click']
    # del shop_service['shop_service_trade']
    # dtrain = pd.merge(dtrain, shop_service, 'left', on=['shop_score_service'])
    # dtest = pd.merge(dtest, shop_service, 'left', on=['shop_score_service'])
    # del shop_service, shop_service_click, shop_service_trade
    #
    # shop_id_trade = dtrain.groupby(['shop_id', 'is_trade']).size().reset_index(name='shop_id_trade')
    # shop_id_trade = shop_id_trade[shop_id_trade['is_trade'] == 1].drop('is_trade', axis=1)
    # shop_id_click = dtrain.groupby(['shop_id']).size().reset_index(name='shop_id_click')
    # shop_id = pd.merge(shop_id_trade, shop_id_click, 'left', on=['shop_id'])
    # shop_id = shop_id.astype(float)
    # shop_id['shop_id_trade_rate'] = shop_id['shop_id_trade'] / shop_id['shop_id_click']
    # dtrain = pd.merge(dtrain, shop_id, 'left', on=['shop_id'])
    # dtest = pd.merge(dtest, shop_id, 'left', on=['shop_id'])
    # del shop_id, shop_id_click, shop_id_trade
    #
    # category
    # predict_category_0_trade = dtrain.groupby(['predict_category_0', 'is_trade']).size().reset_index(
    #     name='predict_category_0_trade')
    # predict_category_0_trade = predict_category_0_trade[predict_category_0_trade['is_trade'] == 1].drop('is_trade',
    #                                                                                                     axis=1)
    # predict_category_0_click = dtrain.groupby(['predict_category_0']).size().reset_index(
    #     name='predict_category_0_click')
    # predict_category_0 = pd.merge(predict_category_0_trade, predict_category_0_click, 'left', on=['predict_category_0'])
    # predict_category_0 = predict_category_0.astype(float)
    # predict_category_0['predict_category_0_trade_rate'] = predict_category_0['predict_category_0_trade'] / \
    #                                                       predict_category_0['predict_category_0_click']
    # dtrain = pd.merge(dtrain, predict_category_0, 'left', on=['predict_category_0'])
    # dtest = pd.merge(dtest, predict_category_0, 'left', on=['predict_category_0'])
    # del predict_category_0, predict_category_0_click, predict_category_0_trade
    #
    # category_1_trade = dtrain.groupby(['category_1', 'is_trade']).size().reset_index(name='category_1_trade')
    # category_1_trade = category_1_trade[category_1_trade['is_trade'] == 1].drop('is_trade', axis=1)
    # category_1_click = dtrain.groupby(['category_1']).size().reset_index(name='category_1_click')
    # category_1 = pd.merge(category_1_trade, category_1_click, 'left', on=['category_1'])
    # category_1 = category_1.astype(float)
    # category_1['category_1_trade_rate'] = category_1['category_1_trade'] / category_1['category_1_click']
    # dtrain = pd.merge(dtrain, category_1, 'left', on=['category_1'])
    # dtest = pd.merge(dtest, category_1, 'left', on=['category_1'])
    # del category_1, category_1_click, category_1_trade

    # context_page_id_trade = dtrain.groupby(['context_page_id', 'is_trade']).size().reset_index(name='context_page_id_trade')
    # context_page_id_trade = context_page_id_trade[context_page_id_trade['is_trade'] == 1].drop('is_trade', axis=1)
    # context_page_id_click = dtrain.groupby(['context_page_id']).size().reset_index(name='context_page_id_click')
    # context_page_id = pd.merge(context_page_id_trade, context_page_id_click, 'left', on=['context_page_id'])
    # context_page_id = context_page_id.astype(float)
    # context_page_id['context_page_id_trade_rate'] = context_page_id['context_page_id_trade'] / context_page_id['context_page_id_click']
    # # del context_page_id['context_page_id_click'], context_page_id['context_page_id_trade']
    # dtrain = pd.merge(dtrain, context_page_id, 'left', on=['context_page_id'])
    # dtest = pd.merge(dtest, context_page_id, 'left', on=['context_page_id'])
    # del context_page_id, context_page_id_click, context_page_id_trade

    # city_trade = dtrain.groupby(['item_city_id', 'is_trade']).size().reset_index(name='city_trade')
    # city_trade = city_trade[city_trade['is_trade'] == 1].drop('is_trade', axis=1)
    # city_click = dtrain.groupby(['item_city_id']).size().reset_index(name='city_click')
    # city = pd.merge(city_trade, city_click, 'left', on=['item_city_id'])
    # city = city.astype(float)
    # city['city_trade_rate'] = city['city_trade'] / city['city_click']
    # dtrain = pd.merge(dtrain, city, 'left', on=['item_city_id'])
    # dtest = pd.merge(dtest, city, 'left', on=['item_city_id'])

    return dtrain, dtest


def binning(col, cut_points, labels=None):

    #Define min and max values:
    minval = col.min()
    maxval = col.max()

    #利用最大值和最小值创建分箱点的列表
    break_points = [minval] + cut_points + [maxval]

    #如果没有标签，则使用默认标签0 ... (n-1)
    if not labels:
        labels = range(len(cut_points)+1)

    #使用pandas的cut功能分箱
    colBin = pd.cut(col,bins=break_points,labels=labels,include_lowest=True)
    return colBin


def add_binning(df):
    # hour
    # df['hour_bin'] = binning(df['hour'], [6, 14, 22], labels=[0, 1, 2, 0])
    df['shop_delivery_bin'] = binning(df['shop_score_delivery'], [0.95, 0.96, 0.97, 0.98, 0.99])
    df['shop_description_bin'] = binning(df['shop_score_description'], [0.95, 0.96, 0.97, 0.98, 0.99])
    df['shop_service_bin'] = binning(df['shop_score_service'], [0.95, 0.96, 0.97, 0.98, 0.99])
    # df['shop_delivery_bin'].fillna(3)
    # del df['shop_score_delivery'], df['shop_score_description']
    return df



def deal_neg1(dtrain, dtest):
    ndtrain = dtrain.replace(-1, np.nan)
    # dtrain = ndtrain.dropna()
    for name in ['item_sales_level', 'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level', 'shop_review_positive_rate',
                 'shop_score_service', 'shop_score_delivery', 'shop_score_description']:
        mode_value = ndtrain.mode()[name][0]
        dtrain[name] = dtrain[name].replace(-1, mode_value)
        dtest[name] = dtest[name].replace(-1, mode_value)
        # print name
    return dtrain, dtest


def data_split():
    print '********************************************'
    print 'data split...'
    print '********************************************\n'

    dtrain = pd.read_csv('../data/round1_ijcai_18_train_20180301.txt', sep=" ")
    dtest = pd.read_csv("../data/round1_ijcai_18_test_b_20180418.txt", sep=" ")

    dtrain = add_date(dtrain)
    dtest = add_date(dtest)
    dtrain = get_prop(dtrain)
    dtest = get_prop(dtest)

    # 18 19 20 21 22 23 24 || 25
    train01 = dtrain[(dtrain['day'] >= 18) & (dtrain['day'] <= 22)]
    train01['day'] = train01['day'].map(lambda x: x - 18)
    test01 = dtrain[dtrain['day'] == 23]
    test01['day'] = test01['day'].map(lambda x: x - 18)

    train02 = dtrain[(dtrain['day'] >= 19) & (dtrain['day'] <= 23)]
    train02['day'] = train02['day'].map(lambda x: x - 19)
    test02 = dtrain[dtrain['day'] == 24]
    test02['day'] = test02['day'].map(lambda x: x - 19)

    train_online = dtrain[(dtrain['day'] >= 20) & (dtrain['day'] <= 24)]
    train_online['day'] = train_online['day'].map(lambda x: x - 20)
    test_online = dtest
    test_online['day'] = test_online['day'].map(lambda x: x - 20)

    # print train1.columns

    train1 = train01.copy()
    train2 = train01.copy()
    train3 = train02.copy()
    train4 = train02.copy()

    test1 = test01.sample(frac=0.7, random_state=13)
    test2 = test01.sample(frac=0.7, random_state=1337)
    test3 = test02.sample(frac=0.7, random_state=13)
    test4 = test02.sample(frac=0.7, random_state=1337)

    dataset = [(train1, test1), (train2, test2), (train3, test3), (train4, test4), (train_online, test_online)]

    for i in range(5):
        dataset[i] = (deal_neg1(dataset[i][0], dataset[i][1]))
        dataset[i] = (add_binning(dataset[i][0]), add_binning(dataset[i][1]))
        # dataset[i] = (add_history(dataset[i][0], dataset[i][1]))

    # train_online, test_online = add_history(train_online, test_online)

    for (train, test) in dataset:
        # del train['day'], test['day']130
        # del train['context_timestamp'], test['context_timestamp']
        le = LabelEncoder()
        l = list(test.columns)
        for i in ['instance_id']:#, 'shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery', 'shop_score_description']:
            l.remove(i)
        for name in l:
            le.fit_transform((list(train[name]) + list(test[name])))
            train[name] = le.transform(train[name])
            test[name] = le.transform(test[name])

    for i, (train, test) in enumerate(dataset[:4]):
        train.to_csv('train%d.csv'%i, columns=train.columns, index=False)
        test.to_csv('test%d.csv'%i, columns=test.columns, index=False)

    dataset[4][0].to_csv('train_online.csv', columns=dataset[4][0].columns, index=False)
    dataset[4][1].to_csv('test_online.csv', columns=dataset[4][1].columns, index=False)

if __name__ == '__main__':
    data_split()