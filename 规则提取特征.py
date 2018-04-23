#coding:utf-8
'''
author:mapodoufu
'''
import time
import pandas as pd
import numpy as np

# 读取数据
def load_dat():
    part_1 = pd.read_csv('./meinian_round1_data_part1_20180408.txt',sep='$')
    part_2 = pd.read_csv('./meinian_round1_data_part2_20180408.txt',sep='$')
    part_1_2 = pd.concat([part_1,part_2])
    part_1_2 = pd.DataFrame(part_1_2).sort_values('vid').reset_index(drop=True)
    begin_time = time.time()
    print('begin')
    # 重复数据的拼接操作
    def merge_table(df):
        df['field_results'] = df['field_results'].astype(str)
        if df.shape[0] > 1:
            merge_df = " ".join(list(df['field_results']))
        else:
            merge_df = df['field_results'].values[0]
        return merge_df
    # 数据简单处理
    print('find_is_copy')
    print(part_1_2.shape)
    is_happen = part_1_2.groupby(['vid','table_id']).size().reset_index()
    # 重塑index用来去重
    is_happen['new_index'] = is_happen['vid'] + '_' + is_happen['table_id']
    is_happen_new = is_happen[is_happen[0]>1]['new_index']

    part_1_2['new_index'] = part_1_2['vid'] + '_' + part_1_2['table_id']

    unique_part = part_1_2[part_1_2['new_index'].isin(list(is_happen_new))]
    unique_part = unique_part.sort_values(['vid','table_id'])
    no_unique_part = part_1_2[~part_1_2['new_index'].isin(list(is_happen_new))]
    print('begin')
    part_1_2_not_unique = unique_part.groupby(['vid','table_id']).apply(merge_table).reset_index()
    part_1_2_not_unique.rename(columns={0:'field_results'},inplace=True)
    print('xxx')
    tmp = pd.concat([part_1_2_not_unique,no_unique_part[['vid','table_id','field_results']]])
    # 行列转换
    print('finish')
    tmp = tmp.pivot(index='vid',values='field_results',columns='table_id')
    tmp.to_csv('./tmp.csv')
    print(tmp.shape)
    print('totle time',time.time() - begin_time)
    return tmp


class reg_class(object):
    def __init__(self, feature_tag):
        self.__feature_tag = feature_tag

    def set_feature_tag(self, feature_tag):
        self.__feature_tag = feature_tag

    def judge(self, x):
        p = 0
        x = str(x)
        for f_tag in self.__feature_tag:
            if f_tag in x:
                p = 1
        return p


def create_regular_features(regular_features, tmp):
    reg_list = []
    for k in regular_features.keys():
        arr = np.zeros([tmp.shape[0], 1])
        tmp_df = pd.DataFrame(arr, columns=[k], index=tmp.index)
        reg_list.append(tmp_df)
    tmp_df = pd.concat(reg_list, axis=1)
    return tmp_df

def set_regular_features(df, reg_dict):
    tmp_new = create_regular_features(reg_dict, df)
    reg_apply = reg_class(reg_dict[list(reg_dict.keys())[0]])
    for i in range(tmp_new.shape[0]):
        for k in reg_dict.keys():
            reg_apply.set_feature_tag(reg_dict[k])
            if (df.iloc[i,:].apply(reg_apply.judge).sum()) > 0:
                tmp_new[k][i] = 1
    arr = tmp_new.iloc[:,-len(reg_dict):].values
    tmp_new = pd.DataFrame(arr, columns=reg_dict.keys(), index=df.index)
    return tmp_new



if __name__ == '__main__':

    regular_features = {
        '脂肪肝': ['脂肪肝'],
        '甲亢': ['甲亢', '甲状腺亢进', '甲状腺功能亢进'],
        '高血压': ['高血压', '血压偏高'],
        '糖尿病': ['糖尿病', '血糖偏高']
    }

    tmp = load_dat()
    tmp_new = set_regular_features(tmp, regular_features)
    tmp_new.to_csv('./regular_feature.csv')

