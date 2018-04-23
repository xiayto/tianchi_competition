#coding:utf-8
'''
author:mapodoufu
'''
import time
import pandas as pd
import numpy as np

# 读取数据
part_1 = pd.read_csv('./input/meinian_round1_data_part1_20180408.txt',sep='$')
part_2 = pd.read_csv('./input/meinian_round1_data_part2_20180408.txt',sep='$')
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

# 删除缺失值过多的特征
def remain_feat(df,thresh=0.9):
    exclude_feats = []
    print('----------移除数据缺失多的字段-----------')
    print('移除之前总的字段数量',len(df.columns))
    num_rows = df.shape[0]
    for c in df.columns:
        num_missing = df[c].isnull().sum()
        if num_missing == 0:
            continue
        missing_percent = num_missing / float(num_rows)
        if missing_percent > thresh:
            exclude_feats.append(c)
    print("移除缺失数据的字段数量: %s" % len(exclude_feats))
    # 保留超过阈值的特征
    feats = []
    for c in df.columns:
        if c not in exclude_feats:
            feats.append(c)
    print('剩余的字段数量',len(feats))
    return feats

feats = remain_feat(tmp, thresh=0.95)

df_remain = tmp[feats]
df_remain.to_csv('./input/df_drop_95.csv')


import numpy as np
# 读取train、test集，清洗train集的label

train_df=pd.read_csv('./input/meinian_round1_train_20180408.csv',sep=',',encoding='gbk')
test_df=pd.read_csv('./input/meinian_round1_test_a_20180409.csv',sep=',',encoding='gbk')

def clean_label(x):
    x=str(x)
    if '+' in x:#16.04++
        i=x.index('+')
        x=x[0:i]
    if '>' in x:#> 11.00
        i=x.index('>')
        x=x[i+1:]
    if len(x.split(sep='.'))>2:#2.2.8
        i=x.rindex('.')
        x=x[0:i]+x[i+1:]
    if '未做' in x or '未查' in x or '弃查' in x or '不查' in x or '弃检' in x:
        x=np.nan
    if str(x).isdigit()==False and len(str(x))>4:
        x=x[0:4]
    return x

def data_clean(df):
    for c in ['收缩压','舒张压','血清甘油三酯','血清高密度脂蛋白','血清低密度脂蛋白']:
        df[c]=df[c].apply(clean_label)
        df[c]=df[c].astype('float64')
    return df

train=data_clean(train_df)

# 合并训练集和测试集方便一起清洗
train.set_index('vid', inplace=True)
test_df.set_index('vid', inplace=True)
train_test_df = pd.concat([train, test_df], axis=0)
print(train_test_df.shape)

train_test_with_feature = pd.concat([train_test_df, df_remain], axis=1, join='inner')
print(train_test_with_feature.shape)
train_test_with_feature.to_csv('./input/train_test_with_feature.csv')

train_nums = train.shape[0]

# 清楚缺失值过多的train集样本
def remain_train(df, train_nums, threshold=0.2):
    exclude_peoples = []
    print('-----移除缺失值过多的人-------')
    print('移除之前总的人数', train_nums)
    for i in range(train_nums):
        if(df.ix[i].count() / df.ix[i].shape[0] < 0.2):
            exclude_peoples.append(df.index[i])
    print('移除人数', len(exclude_peoples))
    df.drop(exclude_peoples, axis=0, inplace=True)
    return df

df_remain_2 = remain_train(train_test_with_feature, train_nums, 0.2)

df_remain_2.to_csv('./input/df_remain_2.csv')


# 找出数值型的特征

def is_num(x):
    try:
        float(x)
        if(x == None):
            x = 0
        elif (x != x):
            x = 0
        else:
            x = 1 
    except:
        x = 0
    return x


def find_nums(df, threshold):
    num_feature = []
    for c in df.columns:
        df_mask = df[c].apply(is_num)
        if(df_mask.sum() / df[c].count() > threshold):
            num_feature.append(c)
    return num_feature

num_features = find_nums(df_remain_2, 0.5)
print(len(num_features))


# 进一步删除缺失值过大的数值型特征 70%以上删除

df_nums = df_remain_2[num_features]

df_nums_remain_feats = remain_feat(df_nums, 0.7)
df_nums_remain = df_nums[df_nums_remain_feats]

df_nums_remain.to_csv('./input/df_nums_remain_70.csv')

# 对数值型特征进行清洗 0表示正常

def clean_num_feature(x):
    x = str(x)
    if ('>' in x):
        while('>' in x):
            i = x.index('>')
            x = x[0:i] + x[i+1:]
    if ('<' in x):
        while('<' in x):
            i = x.index('<')
            x = x[0:i] + x[i+1:]
    if ('=' in x):
        while('=' in x):
            i = x.index('=')
            x = x[0:i] + x[i+1:]
    if ('阴性' in x) or ('正常' in x):
        x = 0
    elif ('过缓' in x):
        x = 50
    elif ('过速' in x):
        x = 105
    elif (' ' in x):
        i = x.index(' ')
        try:
            if(float(x[0:i]) == float(x[0:i])):
                x = float(x[0:i])
            else:
                try:
                    x = float(x[i+1:])
                except:
                    x = np.nan
        except:
            try:
                x = float(x[i+1:])
            except:
                x = np.nan
    elif ('做' in x) or ('查' in x) or ('未' in x) or ('检' in x) or ('见' in x):
        x = np.nan
    elif ('+' in x):
        while('+' in x):
            i = a.index('+')
            a = a[0:i] + a[i+1:]
    elif (x == 'None'):
        x = np.nan
    elif ('次' in x):
        i = x.index('次')
        if(x[i-3] == '1'):
            try:
                x = float(x[i-3:i])
            except:
                x = np.nan
        else:
            try:
                x = float(x[i-2:i])
            except:
                x = np.nan
    elif ('/' in x):
        i = x.index('/')
        if(x[i-3] == '1'):
            try:
                x = float(x[i-3:i])
            except:
                x = np.nan
        else:
            try:
                x = float(x[i-2:i])
            except:
                x = np.nan
    elif ('分' in x):
        i = x.index('分')
        if(x[i-3] == '1'):
            try:
                x = float(x[i-3:i])
            except:
                x = np.nan
        else:
            try:
                x = float(x[i-2:i])
            except:
                x = np.nan
    elif ('-' in x):
        x = np.nan
    try:
        x = float(x)
    except:
        print(x)
    return x

def num_feature_clean(df):
    for c in df.columns:
        df[c]=df[c].apply(clean_num_feature)
    return df

df_nums_remain_clean = num_feature_clean(df_nums_remain)

# 清洗好的数值型数据
df_nums_remain_clean.to_csv('./input/df_nums_remain_clean.csv')


















