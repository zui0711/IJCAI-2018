# coding=utf-8

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/round1_ijcai_18_train_20180301.txt", sep=" ")
dft = pd.read_csv("data/round1_ijcai_18_test_a_20180301.txt", sep=" ")
# print df.dtypes


column_name_total = ['instance_id', 'item_id', 'item_category_list', 'item_property_list', 'item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level', 'user_id', 'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level', 'context_id', 'context_timestamp', 'context_page_id', 'predict_category_property', 'shop_id', 'shop_review_num_level', 'shop_review_positive_rate', 'shop_star_level', 'shop_score_service', 'shop_score_delivery', 'shop_score_description', 'is_trade']

column_name_id = ['instance_id', 'item_id', 'item_brand_id', 'item_city_id', 'user_id', 'user_gender_id', 'user_occupation_id', 'context_id', 'context_page_id', 'shop_id']
column_name_x = ['item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level', 'user_age_level', 'user_star_level', 'context_timestamp', 'shop_review_num_level', 'shop_review_positive_rate', 'shop_star_level', 'shop_score_service', 'shop_score_delivery', 'shop_score_description']

column_name_p = ['item_category_list', 'item_property_list', 'predict_category_property']

column_name_label = 'is_trade'


# print len(column_name_x), len(column_name_p), len(column_name_id)

# df[[x for x in df.columns if x not in column_name_p]].to_excel("aaa.xlsx", index=False)
# df[[x for x in df.columns if x not in column_name_p]].to_csv("aaa.csv", index=False)
# df[[x for x in df.columns if x in column_name_p]].to_excel("aaa.xlsx")

# new_df = df[[x for x in df.columns if x not in column_name_p]].dtypes

# new_df = df[column_name_x]

# for name in column_name_x:
#     fig = plt.figure()
#     d = new_df.sort_values(by=[name],ascending=True)
#
#     plt.scatter(range(len(d)), d[name])
#     plt.title(name)
#     plt.savefig("pic/"+name+".png")
#     plt.close()

# print df.isnull().any()

# print(df["shop_score_delivery"] == -1)

# print(df["shop_score_delivery"].sort_values(ascending=True).head(20))
# print(df["shop_score_description"].sort_values(ascending=True).head(20))

"""
# -1
#
# print("---------------item----------------")
# print(df[df["item_sales_level"]==-1].shape)
#
# print("---------------user---------------")
# print(df[df["user_age_level"]==-1].shape)
# print(df[df["user_star_level"]==-1].shape)
# print(df[(df["user_age_level"]==-1) | (df["user_star_level"]==-1)].shape)
#
# print("---------------shop----------------")
# print(df[df["shop_score_description"]==-1].shape)
# print(df[df["shop_score_delivery"]==-1].shape)
# print(df[df["shop_score_service"]==-1].shape)
# print(df[df["shop_review_positive_rate"]==-1].shape)
# print(df[(df["shop_score_description"]==-1) |
#          (df["shop_score_delivery"]==-1) |
#          (df["shop_score_service"] == -1) |
#          (df["shop_review_positive_rate"] == -1)
#       ].shape)
#
# print("-----------------------------------")
# print(df[(df["item_sales_level"]==-1) | (df["user_age_level"]==-1)].shape)
# print(df[(df["item_sales_level"]==-1) | (df["user_age_level"]==-1) | (df["shop_score_description"]==-1)].shape)

# positive number
# print(sum(df[(df["item_sales_level"]==-1) | (df["user_age_level"]==-1) | (df["shop_score_description"]==-1)]["is_trade"]))
#
# print(df.shape)
# df.drop(df[(df["item_sales_level"]==-1) | (df["user_age_level"]==-1) | (df["shop_score_description"]==-1)].index, inplace=True)
# print(df.shape)
"""

# for name in column_name_id:
#     print(name, df[df[name]==-1].shape)


# # time
# import time
#
# print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(df["context_timestamp"].min())))
# print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(df["context_timestamp"].max())))
# print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(dft["context_timestamp"].min())))
# print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(dft["context_timestamp"].max())))
#
# ndf = df.sort_values(by="context_timestamp", ascending=True)
#
# time_df = pd.DataFrame(columns=["date", "hour", "minute", "second"])
# time_df["date"] = ndf["context_timestamp"].map(lambda x: time.localtime(x).tm_mday)
# time_df["hour"] = ndf["context_timestamp"].map(lambda x: time.localtime(x).tm_hour)
# time_df["minute"] = ndf["context_timestamp"].map(lambda x: time.localtime(x).tm_min)
# time_df["second"] = ndf["context_timestamp"].map(lambda x: time.localtime(x).tm_sec)
"""
# 2018-09-18 00:00:01=========2018-09-24 23:59:47
ndf = pd.concat([ndf, time_df], axis=1)

# print(ndf.head(5))
time_cor = []
print(len(ndf))
for mdate in range(0, 24):
    pos_ad = sum(ndf[ndf["hour"] == mdate]["is_trade"])
    total_ad = len(ndf[ndf["hour"] == mdate])
    perc = float(pos_ad) / total_ad
    # time_cor.append(perc)
    time_cor.append(pos_ad)
    print(pos_ad, total_ad, perc)

plt.bar(range(0, 24), time_cor)
plt.show()
"""
import time
# id
# for name in df.columns:
#     d = df.sort_values(by=name, ascending=True)
#     nd = list(d[name])
#     print(name, len(nd), len(set(nd)))

# print df['shop_score_delivery'].value_counts()
# shop_delivery_trade = df.groupby(['shop_score_delivery', 'is_trade']).size().reset_index(
#     name='shop_delivery_trade')
# shop_delivery_trade = shop_delivery_trade[shop_delivery_trade['is_trade'] == 1].drop('is_trade', axis=1)
# shop_delivery_click = df.groupby(['shop_score_delivery']).size().reset_index(name='shop_delivery_click')
# shop_delivery = pd.merge(shop_delivery_trade, shop_delivery_click, 'left', on=['shop_score_delivery'])
# shop_delivery = shop_delivery.astype(float)
# shop_delivery['shop_delivery_trade_rate'] = shop_delivery['shop_delivery_trade'] / shop_delivery[
#     'shop_delivery_click']
# a = shop_delivery.sort_values(by='shop_score_delivery').drop(shop_delivery['shop_score_delivery'] == -1)
# print a
# plt.plot(a['shop_score_delivery'], a['shop_delivery_trade_rate'])
# plt.show()

# print df['shop_score_description'].value_counts()
# shop_description_trade = df.groupby(['shop_score_description', 'is_trade']).size().reset_index(
#     name='shop_description_trade')
# shop_description_trade = shop_description_trade[shop_description_trade['is_trade'] == 1].drop('is_trade', axis=1)
# shop_description_click = df.groupby(['shop_score_description']).size().reset_index(name='shop_description_click')
# shop_description = pd.merge(shop_description_trade, shop_description_click, 'left', on=['shop_score_description'])
# shop_description = shop_description.astype(float)
# shop_description['shop_description_trade_rate'] = shop_description['shop_description_trade'] / shop_description[
#     'shop_description_click']
# a = shop_description.sort_values(by='shop_score_description').drop(shop_description['shop_score_description'] == -1)
# print a
# plt.plot(a['shop_score_description'], a['shop_description_trade_rate'])
# plt.show()


# print df['shop_score_service'].value_counts()
# shop_service_trade = df.groupby(['shop_score_service', 'is_trade']).size().reset_index(
#     name='shop_service_trade')
# shop_service_trade = shop_service_trade[shop_service_trade['is_trade'] == 1].drop('is_trade', axis=1)
# shop_service_click = df.groupby(['shop_score_service']).size().reset_index(name='shop_service_click')
# shop_service = pd.merge(shop_service_trade, shop_service_click, 'left', on=['shop_score_service'])
# shop_service = shop_service.astype(float)
# shop_service['shop_service_trade_rate'] = shop_service['shop_service_trade'] / shop_service[
#     'shop_service_click']
# a = shop_service.sort_values(by='shop_score_service').drop(shop_service['shop_score_service'] == -1)
# print a
# plt.plot(a['shop_score_service'], a['shop_service_trade_rate'])
# plt.show()

# ndf = df.drop(df['shop_score_delivery'] == -1)

# ndf.boxplot(column='shop_score_delivery')
# ndf['shop_score_delivery'].hist()
# hour['hour_trade_rate'].plot()
# plt.show()
# hour['hour_trade'].plot()
# plt.show()
# hour['hour_click'].plot()
# plt.show()

# fig = plt.figure()

# plt.scatter(ndf["context_timestamp"], ndf["is_trade"])
# plt.title(name)
# plt.show()
# plt.savefig("pic/"+name+".png")
# plt.close()
# #
# for name in dft.columns:
#     print '%s: \t\t\t\t\t%d\t\t\t%d'%(name, len(dft[dft[name] == -1]), len(dft[name]))