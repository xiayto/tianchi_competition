import pandas as pd
import numpy as np
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV
import multiprocessing
from multiprocessing import Process,Queue,Pool
from sklearn.ensemble import RandomForestRegressor


def find_params(para_dict, estimator, x_train, y_train):
    gsearch = GridSearchCV(estimator, param_grid=para_dict, scoring='neg_mean_squared_error',
                           n_jobs=4, iid=False, cv=5)
    gsearch.fit(x_train, y_train)
    return gsearch.best_params_, gsearch.best_score_



def run(x_train, y_train, i, x_predict, blending_i):

    clf = RandomForestRegressor(
            n_estimators=2,             # 学习器个数
            criterion='mse',             # 评价函数
            max_depth=None,              # 最大的树深度，防止过拟合
            min_samples_split=2,         # 根据属性划分节点时，每个划分最少的样本数
            min_samples_leaf=1,          # 最小叶子节点的样本数，防止过拟合
            max_features='auto',         # auto是sqrt(features)还有 log2 和 None可选
            max_leaf_nodes=None,         # 叶子树的最大样本数
            bootstrap=True,              # 有放回的采样
            min_weight_fraction_leaf=0,
            n_jobs=5)                   # 同时用多少个进程训练

    # 1 首先确定迭代次数
    param_test1 = {
        'n_estimators': [i for i in range(100, 201, 20)]
    }
    best_params, best_score = find_params(param_test1, clf, x_train, y_train)
    print('model_rf', i, ':')
    print(best_params, ':best_score:', best_score)
    clf.set_params(n_estimators=best_params['n_estimators'])

    # 2.1 对max_depth 和 min_samples_split 和 min_samples_leaf 进行粗调
    param_test2_1 = {
        'max_depth': [20, 25, 30],
        'min_samples_split' : [10, 25],
        'min_samples_leaf' : [10, 25]
    }
    best_params, best_score = find_params(param_test2_1, clf, x_train, y_train)

    # 2.2 对max_depth 和 min_samples_split 和 min_samples_leaf 进行精调
    max_d = best_params['max_depth']
    min_ss = best_params['min_samples_split']
    min_sl = best_params['min_samples_leaf']
    param_test2_2 = {
        'max_depth': [max_d-2, max_d, max_d+2],
        'min_samples_split': [min_ss-5, min_ss, min_ss+5],
        'min_samples_leaf' : [min_sl-5, min_sl, min_sl+5]
    }
    best_params, best_score = find_params(param_test2_2, clf, x_train, y_train)
    clf.set_params(max_depth=best_params['max_depth'], min_samples_split=best_params['min_samples_split'],
                   min_samples_leaf=best_params['min_samples_leaf'])
    print('model_rf', i, ':')
    print(best_params, ':best_score:', best_score)

    # 3.1 对 max_features 进行调参：
    param_test3_1 = {
        'max_features': [0.5, 0.7, 0.9]
    }
    best_params, best_score = find_params(param_test3_1, clf, x_train, y_train)

    # 3.2 对 max_features 进行精调：
    max_f = best_params['max_features']
    param_test3_2 = {
        'max_features': [max_f-0.1, max_f, max_f+0.1]
    }
    best_params, best_score = find_params(param_test3_2, clf, x_train, y_train)
    clf.set_params(max_features=best_params['max_features'])
    print('model_rf', i, ':')
    print(best_params, ':best_score:', best_score)

    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_predict)
    blending_predict_i = clf.predict(blending_i)

    f = open('./rf_para/model' + str(i) + '.txt', 'w')
    f.write(str(clf.get_params()))
    f.close()

    return y_predict, blending_predict_i



if __name__ == '__main__':
    path_train = './x_train.csv'
    path_predict = './predict.csv'
    path_blending = './x_blending.csv'

    df_train  = pd.read_csv(path_train, sep=',', encoding='gbk', header=None)
    df_predict = pd.read_csv(path_predict, sep=',', encoding='gbk', header=None)
    df_blending = pd.read_csv(path_blending, sep=',', encoding='gbk', header=None)

    x_train = df_train.values[:,1:-5]
    y_train_all = df_train.values[:, -5:]

    x_predict = df_predict.values[:,1:]
    blending_predict = df_blending.values[:,1:-5]

    y_predict = []
    blending_out = []
    for i in range(5):
        y_predict_i, blending_i= run(x_train, y_train_all[:, i], i, x_predict, blending_predict)
        y_predict.append(y_predict_i.reshape((-1,1)))
        blending_out.append(blending_i.reshape((-1,1)))


    y_predict = np.concatenate(y_predict, axis=1)
    y_predict = np.concatenate([df_predict.iloc[:,0].values.reshape((-1,1)), y_predict], axis=1)
    predict = pd.DataFrame(y_predict)
    predict.to_csv('./submit_rf.csv', index=None, header=None)

    blending_out = np.concatenate(blending_out, axis=1)
    blending_out = np.concatenate([df_blending.iloc[:,0].values.reshape((-1,1)), blending_out], axis=1)
    blending = pd.DataFrame(blending_out)
    blending.to_csv('./blending_rf.csv', index=None, header=None)
