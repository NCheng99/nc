import pandas as pd
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import lightgbm as lgb
from lightgbm import LGBMRegressor
import xgboost as xgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold

warnings.simplefilter("ignore")

# 读取训练和测试数据
base_train_df = pd.read_csv('rohlik-orders-forecasting-challenge/train.csv')
base_test_df = pd.read_csv('rohlik-orders-forecasting-challenge/test.csv')

# 提取特征列名（去掉'id'列）
base_features = base_test_df.drop(columns=['id']).columns

# 提取测试数据的id列
test_id = base_test_df['id']

# 合并训练集和测试集以进行统一处理
train_df = pd.concat([base_train_df[base_features], base_train_df['orders']], axis=1)
test_df = base_test_df[base_features]
all_df = pd.concat([train_df, test_df], sort=False).reset_index(drop=True)

# 打印训练和测试数据的信息
print(train_df.info())
print('=' * 60)
print(test_df.info())

print(all_df.info())

# 处理日期特征
date_start = pd.to_datetime(all_df['date'], errors='coerce').min()

date_col = ['date']
for _col in date_col:
    date_col = pd.to_datetime(all_df[_col], errors='coerce')

    # 提取日期特征
    all_df[_col + "_year"] = date_col.dt.year.fillna(-1)
    all_df[_col + "_month"] = date_col.dt.month.fillna(-1)
    all_df[_col + "_day"] = date_col.dt.day.fillna(-1)
    all_df[_col + "_day_of_week"] = date_col.dt.dayofweek.fillna(-1)
    all_df[_col + "_week_of_year"] = date_col.dt.isocalendar().week.fillna(-1)
    all_df[_col + "_num"] = (date_col - date_start).dt.days.fillna(-1)
    all_df[_col + "_day_of_year"] = date_col.dt.dayofyear.fillna(-1)

    # 处理闰年影响
    all_df[_col + "_day_of_year"] = np.where((all_df[_col + "_year"] % 4 == 0) & (all_df[_col + "_month"] > 2),
                                             all_df[_col + "_day_of_year"] - 1, all_df[_col + "_day_of_year"])

    # 提取其他日期相关特征
    all_df[_col + "_quarter"] = date_col.dt.quarter.fillna(-1)
    all_df[_col + "_is_month_start"] = date_col.dt.is_month_start.astype(int).fillna(-1)
    all_df[_col + "_is_month_end"] = date_col.dt.is_month_end.astype(int).fillna(-1)
    all_df[_col + "_is_quarter_start"] = date_col.dt.is_quarter_start.astype(int).fillna(-1)
    all_df[_col + "_is_quarter_end"] = date_col.dt.is_quarter_end.astype(int).fillna(-1)

# 转换日期为周期性特征
all_df['month_sin'] = all_df['date_month'] * np.sin(2 * pi * all_df['date_month'])
all_df['month_cos'] = all_df['date_month'] * np.cos(2 * pi * all_df['date_month'])
all_df['day_sin'] = all_df['date_day'] * np.sin(2 * pi * all_df['date_day'])
all_df['day_cos'] = all_df['date_day'] * np.cos(2 * pi * all_df['date_day'])
all_df['year_sin'] = np.sin(2 * pi * all_df["date_day_of_year"])
all_df['year_cos'] = np.cos(2 * pi * all_df['date_day_of_year'])


# 填充假期名称的缺失值，并进行独热编码
all_df['holiday_name'].fillna('None', inplace=True)
enc = OneHotEncoder(sparse_output=False)
holiday_encoded = enc.fit_transform(all_df[['holiday_name']])
encoded_df = pd.DataFrame(holiday_encoded, columns=enc.get_feature_names_out(['holiday_name']))
all_df = pd.concat([all_df, encoded_df], axis=1)
all_df = all_df.drop('holiday_name', axis=1)

# 对仓库列进行标签编码
le = preprocessing.LabelEncoder()
all_df['warehouse'] = le.fit_transform(all_df['warehouse'])

# 创建假期前后状态特征
all_df['holiday_before'] = all_df['holiday'].shift(1).fillna(0).astype(int)
all_df['holiday_after'] = all_df['holiday'].shift(-1).fillna(0).astype(int)

# 将数据框分回训练集和测试集
train_df_le = all_df[~all_df['orders'].isnull()]
test_df_le = all_df[all_df['orders'].isnull()]
train_df_le = train_df_le.drop(columns=['date'], axis=1)
test_df_le = test_df_le.drop(columns=['date'], axis=1)

print(all_df.head())

# 设置随机种子
# 设置随机种子，以确保结果的可复现性
random_seed = 777

# 拆分特征和目标变量，将特征变量存入X，目标变量存入y
X = train_df_le.drop(columns=['orders'])  # 'orders' 列为目标变量
y = train_df_le['orders']

# 进一步拆分数据为训练集和验证集，20%的数据用作验证集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

# 定义交叉验证参数，使用10折交叉验证
n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

# 初始化堆叠训练和测试的结果矩阵，用于存储各个基模型的预测结果
stacking_train = np.zeros((X_train.shape[0], 7))  # 训练集预测结果
stacking_test = np.zeros((X_test.shape[0], 7))    # 测试集预测结果

# 初始化基模型，包括 LightGBM、XGBoost、CatBoost、随机森林、逻辑回归、AdaBoost、决策树和梯度提升树
lgb_model = lgb.LGBMRegressor(verbosity=-1,random_state=random_seed)
xgb_model = xgb.XGBRegressor(random_state=random_seed)
cat_model = CatBoostRegressor(silent=True, random_state=random_seed)
rf_model = RandomForestRegressor(random_state=random_seed)
ad_model = AdaBoostRegressor(random_state=random_seed)
dt_model = DecisionTreeRegressor(random_state=random_seed)
gb_model = GradientBoostingRegressor(random_state=random_seed)

# 使用交叉验证训练每个基模型
for train_idx, val_idx in kf.split(X_train):
    # 使用当前折的索引划分训练集和验证集
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # 训练每个基模型
    lgb_model.fit(X_tr, y_tr)
    xgb_model.fit(X_tr, y_tr)
    cat_model.fit(X_tr, y_tr)
    rf_model.fit(X_tr, y_tr)
    ad_model.fit(X_tr, y_tr)
    dt_model.fit(X_tr, y_tr)
    gb_model.fit(X_tr, y_tr)

    # 在验证集上进行预测并保存到堆叠训练矩阵
    stacking_train[val_idx, 0] = lgb_model.predict(X_val)
    stacking_train[val_idx, 1] = xgb_model.predict(X_val)
    stacking_train[val_idx, 2] = cat_model.predict(X_val)
    stacking_train[val_idx, 3] = rf_model.predict(X_val)
    stacking_train[val_idx, 4] = ad_model.predict(X_val)
    stacking_train[val_idx, 5] = dt_model.predict(X_val)
    stacking_train[val_idx, 6] = gb_model.predict(X_val)

    # 在测试集上进行预测并保存到堆叠测试矩阵，这里使用了分数的平均值
    stacking_test[:, 0] += lgb_model.predict(X_test) / n_splits
    stacking_test[:, 1] += xgb_model.predict(X_test) / n_splits
    stacking_test[:, 2] += cat_model.predict(X_test) / n_splits
    stacking_test[:, 3] += rf_model.predict(X_test) / n_splits
    stacking_test[:, 4] += ad_model.predict(X_test) / n_splits
    stacking_test[:, 5] += dt_model.predict(X_test) / n_splits
    stacking_test[:, 6] += gb_model.predict(X_test) / n_splits

# 初始化元模型（第二层模型），用于堆叠各个基模型的预测结果
meta_model_1 = LGBMRegressor(n_estimators=150, num_leaves=15, learning_rate=0.05,
                             colsample_bytree=0.6, lambda_l1=0.2, lambda_l2=0.2, random_state=random_seed)
meta_model_2 = CatBoostRegressor(verbose=0, random_state=random_seed)
meta_model_3 = XGBRegressor(random_state=random_seed)

# 使用堆叠的训练数据训练元模型
meta_model_1.fit(stacking_train, y_train)
meta_model_2.fit(stacking_train, y_train)
meta_model_3.fit(stacking_train, y_train)

# 预测测试集的结果
meta_pred_1 = meta_model_1.predict(stacking_test)
meta_pred_2 = meta_model_2.predict(stacking_test)
meta_pred_3 = meta_model_3.predict(stacking_test)

# 对测试数据进行预测（未使用交叉验证）
test_df_le = test_df_le.drop(columns=['orders'])  # 删除目标列，以获得特征

# 各基模型在测试数据上的预测
lgb_pred_test = lgb_model.predict(test_df_le)
xgb_pred_test = xgb_model.predict(test_df_le)
cat_pred_test = cat_model.predict(test_df_le)
rf_pred_test = rf_model.predict(test_df_le)
ad_pred_test = ad_model.predict(test_df_le)
dt_pred_test = dt_model.predict(test_df_le)
gb_pred_test = gb_model.predict(test_df_le)

# 将各个基模型的预测结果组合到堆叠测试矩阵中
stacking_test_df_le = np.vstack([lgb_pred_test, xgb_pred_test, cat_pred_test, rf_pred_test, ad_pred_test, dt_pred_test, gb_pred_test]).T

# 使用元模型预测最终的结果
submit_pred_1 = meta_model_1.predict(stacking_test_df_le)
submit_pred_2 = meta_model_2.predict(stacking_test_df_le)
submit_pred_3 = meta_model_3.predict(stacking_test_df_le)

# 定义模型权重，用于加权组合最终预测结果
weights = {
    'cat_test_preds': 1/3,
    'lgb_test_preds': 1/3,
    'xgb_test_preds': 1/3,
}

# 根据权重对各元模型的预测结果进行加权平均
cat_test_preds_weighted = submit_pred_2 * weights['cat_test_preds']
lgb_test_preds_weighted = submit_pred_1 * weights['lgb_test_preds']
xgb_test_preds_weighted = submit_pred_3 * weights['xgb_test_preds']
submit_pred = cat_test_preds_weighted + lgb_test_preds_weighted + xgb_test_preds_weighted

# 创建提交文件，包含测试集ID和预测结果
submission = pd.DataFrame({
    'id': test_id,
    'Target': submit_pred
})

# 保存提交文件到CSV格式
submission.to_csv('submission.csv', index=False)

# 打印提交文件内容
print(submission)
