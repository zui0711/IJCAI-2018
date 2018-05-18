# coding=utf-8

def list_sub(a, b):
    list_new = []
    for n in a:
        if n not in b:
            list_new.append(n)
    return list_new

f = open("data/round1_ijcai_18_train_20180301.txt")

# f_test = open("round1_ijcai_18_test_a_20180301.txt")

field = f.readline().split()
context = f.readlines()

column_name = ['instance_id', 'item_id', 'item_category_list', 'item_property_list', 'item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level', 'user_id', 'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level', 'context_id', 'context_timestamp', 'context_page_id', 'predict_category_property', 'shop_id', 'shop_review_num_level', 'shop_review_positive_rate', 'shop_star_level', 'shop_score_service', 'shop_score_delivery', 'shop_score_description', 'is_trade']

label_column = ['is_trade']
exclude_column_p = ['item_category_list', 'item_property_list', 'predict_category_property']
exclude_column_id = ['instance_id', 'item_id', 'item_brand_id', 'item_city_id', 'user_id', 'user_gender_id', 'user_occupation_id', 'context_id', 'context_page_id', 'shop_id']

print(len(field), field)

value = []
for line in context:
    v = []
    for i, v0 in enumerate(line.split()):
        # print(i, v0)
        # print(switch_case(i, v0))
        if column_name[i] in exclude_column_p:
            continue
        else:
            v.append(v0)
    value.append(v)


# print(value)

# f_save1 = open("data/save_value", "w")
# for i in value:
#     for j in i:
#         f_save1.write(j+", ")
#     f_save1.write("\n")

import pandas as pd
from pandas.core.frame import DataFrame

column_name = list_sub(column_name, exclude_column_p)

data = DataFrame(value, columns=column_name)
print(data.shape)

# data = data.dropna()
# print(data.shape)

# save label
# data[label_column].to_csv(open("data_csv/label.csv", "w"), columns=label_column, index=False)

# save id data
# data[exclude_column_id].to_csv(open("data_csv/id.csv", "w"), columns=exclude_column_id, index=False)

# save data exclude id

column_name = list_sub(column_name, label_column)
# data.to_csv(open("data_csv/data.csv", "w"), columns=column_name, index=False)
print column_name
column_name = list_sub(column_name, exclude_column_id)
print column_name
# data[column_name].to_csv(open("data_csv/simple_info.csv", "w"), columns=column_name, index=False)

#
# def null_count(column):
#     column_null = pd.isnull(column)
#     null = column[column_null == True]
#     return len(null)
#
# for i in data:
#     print null_count(i)
