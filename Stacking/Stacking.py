import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
import warnings
from sklearn.model_selection import cross_val_predict

# 读取数据
train_df = pd.read_csv('../output/WT/测试集A/train.csv')
test_df = pd.read_csv('../output/WT/测试集B/test_b_x.csv')

# 保留测试集的 user_id 列
test_user_ids = test_df['user_id']

# 合并数据
all_df = pd.concat([train_df, test_df], sort=False).reset_index(drop=True)

all_df = all_df.drop(columns=['user_id'], errors='ignore')

# 将合约-1改为0，表示为没有合约
all_df['term_cont_plan_months'] = all_df['term_cont_plan_months'].replace(-1, 0)

# 将KB单位转为MB
all_df['free_app_flux_m3'] = all_df['free_app_flux_m3'] / 1024

# 对curmon_acum_qty按star_level分组补值
all_df['curmon_acum_qty'] = all_df.groupby('star_level')['curmon_acum_qty'].transform(
    lambda x: x.fillna(x.median()))

mean_by_star_level = all_df.groupby('star_level')['curmon_acum_qty'].mean()

print(mean_by_star_level)


def fill_missing_values(row):
    if pd.isna(row['curmon_acum_qty']):  # 如果 curmon_acum_qty 为空
        return mean_by_star_level.get(row['star_level'], np.nan)  # 使用对应 star_level 的均值
    return row['curmon_acum_qty']  # 否则保持原值


all_df['curmon_acum_qty'] = all_df.apply(fill_missing_values, axis=1)

# 要插值的列
columns_to_interpolate = [
    'm1_comm_days', 'm2_comm_days', 'm3_comm_days',
    'm1_calling_cnt', 'm2_calling_cnt', 'm3_calling_cnt'
]

# 对这些列进行补零操作（即将 NaN 值替换为 0）
all_df[columns_to_interpolate] = all_df[columns_to_interpolate].fillna(0)

columns_to_fill = [
    'm1_hot_app_flow', 'm2_hot_app_flow', 'm3_hot_app_flow'
]

all_df[columns_to_fill] = all_df[columns_to_fill].fillna(all_df[columns_to_fill].median())


# 定义分箱的函数
def binning_and_labeling(df, column_name, bin_interval, max_value, bin_label_prefix):
    """
    对指定的列进行分箱，并生成对应的标签和数值编码。

    参数:
    df: pd.DataFrame 数据框
    column_name: str 需要分箱的列名
    bin_interval: int 分箱间隔
    max_value: int 分箱的最大值，超过该值的归为一个组
    bin_label_prefix: str 标签前缀，用于标识分箱结果

    返回: 修改后的数据框
    """
    # 定义分箱区间
    bins = list(range(0, max_value, bin_interval))
    bins.append(np.inf)  # 大于最大值的归为一组

    # 定义标签
    labels = [f"[{bins[i]}, {bins[i + 1]})" for i in range(len(bins) - 1)]

    # 对指定列进行分箱操作
    df[f'binned_{column_name}'] = pd.cut(df[column_name], bins=bins, labels=labels, right=False)

    # 将 Categorical 类型转换为数值型
    df[f'binned_{column_name}_label'] = df[f'binned_{column_name}'].cat.codes

    return df


# 定义4000为间隔的列，最大值为40000
columns_4000 = ['free_app_flux_m3']
for col in columns_4000:
    all_df = binning_and_labeling(all_df, col, 4000, 40000, bin_label_prefix='flux')

columns_30 = ['domestic_roam_call_count_m3', 'contact_avg_called_cnt_m3', 'contact_avg_call_cnt_m3']
for col in columns_30:
    all_df = binning_and_labeling(all_df, col, 30, 300, bin_label_prefix='calling')

columns_20 = ['bw_contact_count_m3', ]
for col in columns_20:
    all_df = binning_and_labeling(all_df, col, 20, 200, bin_label_prefix='bw')

# 打印统计结果
for col in columns_4000 + columns_30 + columns_20:
    grouped = all_df.groupby(f'binned_{col}')[col]
    print(f"\n分箱统计结果 - {col}:")
    for name, group in grouped:
        print(f"箱 {name}:")
        print(f"行数量: {group.count()}")
        print(f"最大值: {group.max()}")
        print(f"最小值: {group.min()}")
        print(f"平均值: {group.mean():.2f}")
        print(f"中位数: {group.median():.2f}")
        print("----------------------------")

# 删除分箱列，保留数值编码
all_df = all_df.drop(columns=[f'binned_{col}' for col in columns_4000 + columns_30 + columns_20])

def add_new_features(df):
    # 手动新增的新特征

    # 3月联系人被叫通话次数与3月联系人通话次数的比率
    df['contact_avg_called/contact_avg_call'] = (df['contact_avg_called_cnt_m3'] /
                                                 df['contact_avg_call_cnt_m3'].replace(0, np.nan)).fillna(0).astype(
        'float32')

    # 1-3月缴费金额总和
    df['m1_m2_m3_owe_fee_sum'] = df[['m1_owe_fee', 'm2_owe_fee', 'm3_owe_fee']].sum(axis=1).astype('float32')

    # 1-3月缴费金额平均值
    df['m1_m2_m3_owe_fee_average'] = df[['m1_owe_fee', 'm2_owe_fee', 'm3_owe_fee']].mean(axis=1).astype('float32')

    # 网龄/年龄比率
    df['innet_months_per_age'] = (df['innet_months'] / df['age'].replace(0, np.nan)).fillna(0).astype('float32')

    # 每星级所需网龄均值
    df['innet_months_per_star_level'] = (df['innet_months'] / df['star_level'].replace(0, np.nan)).fillna(0).astype(
        'float32')

    # 对新增的特征进行取整
    new_features = [
        'contact_avg_called/contact_avg_call', 'm1_m2_m3_owe_fee_sum', 'm1_m2_m3_owe_fee_average',
        'innet_months_per_age', 'innet_months_per_star_level']

    df[new_features] = df[new_features].round(2)

    return df


# 添加新特征
all_df = add_new_features(all_df)

# 填充 NaN 值为 0 的列
columns_to_fill_zero = [
    'innet_months_per_age',
    'innet_months_per_star_level'
]

all_df[columns_to_fill_zero] = all_df[columns_to_fill_zero].fillna(0)


def convert_dtype(df):
    # 将所有 float64 列转换为 float32
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')

    # 将所有 int64 列转换为 int32
    int_cols = df.select_dtypes(include=['int64']).columns
    df[int_cols] = df[int_cols].astype('int32')

    return df


# 假设 all_df 是你的数据框
all_df = convert_dtype(all_df)

print(all_df.info())

all_df.to_csv('../output/填补后的数据.csv', index=False)

random_seed = 777

# 数据集划分
train_df_le = all_df[~all_df['label'].isnull()]  # 训练集（label不为空）
test_df_le = all_df[all_df['label'].isnull()]  # 测试集（label为空）

# 特征和目标变量
X_train = train_df_le.drop(columns=['label'])
y_train = train_df_le['label']
X_test = test_df_le.drop(columns=['label'])  # 测试集没有label，只保留特征

# 定义分层K折交叉验证，使用10折交叉验证
n_splits = 5

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

# 初始化堆叠训练和测试的结果矩阵，用于存储各个基模型的预测结果
stacking_train = np.zeros((X_train.shape[0], 4))  # 训练集预测结果
stacking_test = np.zeros((X_test.shape[0], 4))  # 测试集预测结果

# 初始化基模型，包括 LightGBM、XGBoost、CatBoost、随机森林
lgb_model = lgb.LGBMClassifier(
    num_leaves=168,  # 替换后的叶子节点数
    learning_rate=0.06759547092633311,  # 替换后的学习率
    n_estimators=198,  # 替换后的树的数量
    max_depth=14,  # 替换后的最大深度
    subsample=0.5752001789402736,  # 替换后的子样本比例
    colsample_bytree=0.7461114915303204,  # 替换后的列采样比例
    min_child_samples=16,  # 替换后的叶子节点的最小样本数
    reg_alpha=0.12050022174434907,  # 替换后的L1正则化系数
    reg_lambda=16.710194869890913,  # 替换后的L2正则化系数
    verbosity=-1,
    random_state=random_seed,
    num_threads=10
)

# XGBClassifier模型，应用超参数优化结果
xgb_model = xgb.XGBClassifier(
    max_depth=10,  # 替换为最佳值
    learning_rate=0.09920214024455251,  # 替换为最佳值
    n_estimators=219,  # 替换为最佳值
    subsample=0.7257671086056524,  # 替换为最佳值
    colsample_bytree=0.9213399633935782,  # 替换为最佳值
    gamma=0.09038839426181743,  # 替换为最佳值
    scale_pos_weight=2.806313089968987,  # 替换为最佳值
    min_child_weight=4,  # 替换为最佳值
    random_state=random_seed,
    n_jobs=10)

# CatBoostClassifier模型，应用超参数优化结果
cat_model = CatBoostClassifier(
    iterations=290,  # 替换为最佳值
    depth=10,  # 替换为最佳值
    learning_rate=0.08260556789398812,  # 替换为最佳值
    l2_leaf_reg=2.3048131488661627,  # 替换为最佳值
    border_count=227,  # 替换为最佳值
    bagging_temperature=0.4200025807500587,  # 替换为最佳值
    auto_class_weights=None,  # 替换为最佳值
    silent=True,  # 保持静默模式
    random_state=random_seed, thread_count=10  # 随机种子
)

# RandomForestClassifier模型，应用超参数优化结果
rf_model = RandomForestClassifier(
    n_estimators=110,  # 替换为最佳值
    max_depth=19,  # 替换为最佳值
    min_samples_split=19,  # 替换为最佳值
    min_samples_leaf=4,  # 替换为最佳值
    max_features=0.6818979353230743,  # 替换为最佳值
    bootstrap=True,  # 保持默认值
    max_samples=0.6927402808115899,  # 替换为最佳值
    class_weight='balanced',  # 替换为最佳值
    random_state=random_seed,
    n_jobs=10  # 随机种子
)

# 初始化列表用于存储每个模型的指标
accuracy_lgb_list = []
accuracy_xgb_list = []
accuracy_cat_list = []
accuracy_rf_list = []

auc_lgb_list = []
auc_xgb_list = []
auc_cat_list = []
auc_rf_list = []

f1_lgb_list = []
f1_xgb_list = []
f1_cat_list = []
f1_rf_list = []

# 使用分层交叉验证训练每个基模型
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    # 使用当前折的索引划分训练集和验证集
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # 训练每个基模型
    lgb_model.fit(X_tr, y_tr)
    xgb_model.fit(X_tr, y_tr)
    cat_model.fit(X_tr, y_tr)
    rf_model.fit(X_tr, y_tr)

    # 在验证集上进行预测（使用预测概率的第二列，即正类的概率）
    y_val_pred_lgb = lgb_model.predict_proba(X_val)[:, 1]
    y_val_pred_xgb = xgb_model.predict_proba(X_val)[:, 1]
    y_val_pred_cat = cat_model.predict_proba(X_val)[:, 1]
    y_val_pred_rf = rf_model.predict_proba(X_val)[:, 1]

    # 在验证集上保存预测到堆叠训练矩阵
    stacking_train[val_idx, 0] = y_val_pred_lgb
    stacking_train[val_idx, 1] = y_val_pred_xgb
    stacking_train[val_idx, 2] = y_val_pred_cat
    stacking_train[val_idx, 3] = y_val_pred_rf

    # 在测试集上进行预测，并将结果平均存储
    stacking_test[:, 0] += lgb_model.predict_proba(X_test)[:, 1] / n_splits
    stacking_test[:, 1] += xgb_model.predict_proba(X_test)[:, 1] / n_splits
    stacking_test[:, 2] += cat_model.predict_proba(X_test)[:, 1] / n_splits
    stacking_test[:, 3] += rf_model.predict_proba(X_test)[:, 1] / n_splits

    # 将预测的概率转换为二元分类的标签
    y_val_pred_class_lgb = (y_val_pred_lgb > 0.5).astype(int)
    y_val_pred_class_xgb = (y_val_pred_xgb > 0.5).astype(int)
    y_val_pred_class_cat = (y_val_pred_cat > 0.5).astype(int)
    y_val_pred_class_rf = (y_val_pred_rf > 0.5).astype(int)

    # 计算评估指标
    accuracy_lgb = accuracy_score(y_val, y_val_pred_class_lgb)
    accuracy_xgb = accuracy_score(y_val, y_val_pred_class_xgb)
    accuracy_cat = accuracy_score(y_val, y_val_pred_class_cat)
    accuracy_rf = accuracy_score(y_val, y_val_pred_class_rf)

    auc_lgb = roc_auc_score(y_val, y_val_pred_lgb)
    auc_xgb = roc_auc_score(y_val, y_val_pred_xgb)
    auc_cat = roc_auc_score(y_val, y_val_pred_cat)
    auc_rf = roc_auc_score(y_val, y_val_pred_rf)

    f1_lgb = f1_score(y_val, y_val_pred_class_lgb)
    f1_xgb = f1_score(y_val, y_val_pred_class_xgb)
    f1_cat = f1_score(y_val, y_val_pred_class_cat)
    f1_rf = f1_score(y_val, y_val_pred_class_rf)

    # 保存每折的指标
    accuracy_lgb_list.append(accuracy_lgb)
    accuracy_xgb_list.append(accuracy_xgb)
    accuracy_cat_list.append(accuracy_cat)
    accuracy_rf_list.append(accuracy_rf)

    auc_lgb_list.append(auc_lgb)
    auc_xgb_list.append(auc_xgb)
    auc_cat_list.append(auc_cat)
    auc_rf_list.append(auc_rf)

    f1_lgb_list.append(f1_lgb)
    f1_xgb_list.append(f1_xgb)
    f1_cat_list.append(f1_cat)
    f1_rf_list.append(f1_rf)

    # 打印每个折的评估指标
    print(f"Fold {fold} - LGB: Accuracy={accuracy_lgb:.4f}, AUC={auc_lgb:.4f}, F1={f1_lgb:.4f}")
    print(f"Fold {fold} - XGB: Accuracy={accuracy_xgb:.4f}, AUC={auc_xgb:.4f}, F1={f1_xgb:.4f}")
    print(f"Fold {fold} - Cat: Accuracy={accuracy_cat:.4f}, AUC={auc_cat:.4f}, F1={f1_cat:.4f}")
    print(f"Fold {fold} - RF: Accuracy={accuracy_rf:.4f}, AUC={auc_rf:.4f}, F1={f1_rf:.4f}")

# 计算平均指标
print(
    f"Average LGB: Accuracy={np.mean(accuracy_lgb_list):.4f}, AUC={np.mean(auc_lgb_list):.4f}, F1={np.mean(f1_lgb_list):.4f}")
print(
    f"Average XGB: Accuracy={np.mean(accuracy_xgb_list):.4f}, AUC={np.mean(auc_xgb_list):.4f}, F1={np.mean(f1_xgb_list):.4f}")
print(
    f"Average Cat: Accuracy={np.mean(accuracy_cat_list):.4f}, AUC={np.mean(auc_cat_list):.4f}, F1={np.mean(f1_cat_list):.4f}")
print(
    f"Average RF: Accuracy={np.mean(accuracy_rf_list):.4f}, AUC={np.mean(auc_rf_list):.4f}, F1={np.mean(f1_rf_list):.4f}")

# 结果展示
print("Stacking train matrix:")
print(stacking_train)
print("Stacking test matrix:")
print(stacking_test)

# 将 stacking_train 转换为 DataFrame
df_stacking_train = pd.DataFrame(stacking_train)
df_stacking_test = pd.DataFrame(stacking_test)

# 保存为 CSV 文件
df_stacking_train.to_csv('../output/stacking_train.csv', index=False)  # 保存训练集
df_stacking_test.to_csv('../output/stacking_test.csv', index=False)  # 保存测试集

print("stacking_train 和 stacking_test 已成功保存为 CSV 文件")

# 初始化元模型（第二层模型），用于堆叠各个基模型的预测结果
# meta_model_1 = LGBMClassifier(
#     n_estimators=103,
#     num_leaves=102,
#     learning_rate=0.07233607701185306,
#     colsample_bytree=0.7164398417312575,
#     reg_alpha=57.27633439067915,
#     reg_lambda=0.9458299902809071,
#     max_depth=4,
#     subsample=0.7358275665437826,
#     min_child_samples=16,
#     random_state=random_seed
# )

# meta_model_2 = CatBoostClassifier(
#     iterations=281,
#     depth=10,
#     learning_rate=0.013528432244437255,
#     l2_leaf_reg=11.015573362324071,
#     border_count=90,
#     bagging_temperature=0.5982808019529782,
#     auto_class_weights=None,
#     verbose=0,
#     random_state=random_seed
# )

# meta_model_3 = XGBClassifier(
#     max_depth=3,
#     learning_rate=0.08807511105554496,
#     n_estimators=142,
#     subsample=0.6861946715661591,
#     colsample_bytree=0.6922518580143794,
#     gamma=0.00011154625144572948,
#     scale_pos_weight=1.3474891867938368,
#     min_child_weight=8,
#     random_state=random_seed
# )

meta_model_1 = LGBMClassifier(
    n_estimators=283,  # 从优化结果中获取
    num_leaves=164,  # 从优化结果中获取
    learning_rate=0.05244956785774849,  # 从优化结果中获取
    colsample_bytree=0.8377706781140973,  # 从优化结果中获取
    reg_alpha=1.1878997280675803,  # 从优化结果中获取
    reg_lambda=3.021538707442974,  # 从优化结果中获取
    max_depth=3,  # 从优化结果中获取
    subsample=0.7923464534667846,  # 从优化结果中获取
    min_child_samples=8,  # 从优化结果中获取
    random_state=random_seed
)

# 更新后的 CatBoostClassifier
meta_model_2 = CatBoostClassifier(
    iterations=102,  # 从优化结果中获取
    depth=3,  # 从优化结果中获取
    learning_rate=0.0308096807443678,  # 从优化结果中获取
    l2_leaf_reg=0.018633974188556533,  # 从优化结果中获取
    bagging_temperature=0.10192692802231151,  # 从优化结果中获取
    scale_pos_weight=7.162879025079999,  # 从优化结果中获取
    random_strength=5.877819563735475,  # 从优化结果中获取
    verbose=0,  # 不打印训练信息
    random_state=random_seed
)

# 更新后的 XGBClassifier
meta_model_3 = XGBClassifier(
    max_depth=6,  # 从优化结果中获取
    learning_rate=0.0027262791199156476,  # 从优化结果中获取
    n_estimators=174,  # 从优化结果中获取
    subsample=0.5326410733091405,  # 从优化结果中获取
    colsample_bytree=0.7476897519228524,  # 从优化结果中获取
    gamma=0.0005425644974349196,  # 从优化结果中获取
    scale_pos_weight=7.451783784935816,  # 从优化结果中获取
    min_child_weight=6,  # 从优化结果中获取
    random_state=random_seed
)

# 使用堆叠的训练数据训练元模型
meta_model_1.fit(stacking_train, y_train)
meta_model_2.fit(stacking_train, y_train)
meta_model_3.fit(stacking_train, y_train)

# 元模型 1 的预测概率
meta_predictions_1 = cross_val_predict(meta_model_1, stacking_train, y_train, cv=skf, method='predict_proba')[:, 1]
# 元模型 2 的预测概率
meta_predictions_2 = cross_val_predict(meta_model_2, stacking_train, y_train, cv=skf, method='predict_proba')[:, 1]
# 元模型 3 的预测概率
meta_predictions_3 = cross_val_predict(meta_model_3, stacking_train, y_train, cv=skf, method='predict_proba')[:, 1]

# 设定阈值范围
thresholds = np.arange(0.3, 0.61, 0.01)  # 从0.4到0.6，间隔为0.01

# 找到最佳阈值和相应的AUC
best_auc_1 = 0
best_threshold_1 = 0.5

best_auc_2 = 0
best_threshold_2 = 0.5

best_auc_3 = 0
best_threshold_3 = 0.5

for threshold in thresholds:
    # 根据当前阈值生成预测
    predictions_1 = (meta_predictions_1 >= threshold).astype(int)
    predictions_2 = (meta_predictions_2 >= threshold).astype(int)
    predictions_3 = (meta_predictions_3 >= threshold).astype(int)

    # 计算AUC
    auc_1 = roc_auc_score(y_train, predictions_1)
    auc_2 = roc_auc_score(y_train, predictions_2)
    auc_3 = roc_auc_score(y_train, predictions_3)

    # 打印当前阈值下的AUC
    print(
        f'Threshold: {threshold:.2f} - AUC for Meta Model 1: {auc_1:.4f}, AUC for Meta Model 2: {auc_2:.4f}, AUC for Meta Model 3: {auc_3:.4f}')

    # 更新最佳阈值和AUC
    if auc_1 > best_auc_1:
        best_auc_1 = auc_1
        best_threshold_1 = threshold

    if auc_2 > best_auc_2:
        best_auc_2 = auc_2
        best_threshold_2 = threshold

    if auc_3 > best_auc_3:
        best_auc_3 = auc_3
        best_threshold_3 = threshold

# 输出最佳阈值和相应的AUC
print(f'Best threshold for Meta Model 1: {best_threshold_1}, AUC: {best_auc_1:.4f}')
print(f'Best threshold for Meta Model 2: {best_threshold_2}, AUC: {best_auc_2:.4f}')
print(f'Best threshold for Meta Model 3: {best_threshold_3}, AUC: {best_auc_3:.4f}')

# 预测测试集的结果
meta_pred_1 = meta_model_1.predict_proba(stacking_test)[:, 1]
meta_pred_2 = meta_model_2.predict_proba(stacking_test)[:, 1]
meta_pred_3 = meta_model_3.predict_proba(stacking_test)[:, 1]

# 各基模型在测试数据上的预测
lgb_pred_test = lgb_model.predict_proba(X_test)[:, 1]
xgb_pred_test = xgb_model.predict_proba(X_test)[:, 1]
cat_pred_test = cat_model.predict_proba(X_test)[:, 1]
rf_pred_test = rf_model.predict_proba(X_test)[:, 1]

# 将各个基模型的预测结果组合到堆叠测试矩阵中
stacking_test_df_le = np.vstack([lgb_pred_test, xgb_pred_test, cat_pred_test, rf_pred_test]).T

# 使用元模型预测最终的结果
submit_pred_1 = meta_model_1.predict_proba(stacking_test_df_le)[:, 1]
submit_pred_2 = meta_model_2.predict_proba(stacking_test_df_le)[:, 1]
submit_pred_3 = meta_model_3.predict_proba(stacking_test_df_le)[:, 1]

# 定义模型权重，用于加权组合最终预测结果
weights = {
    'cat_test_preds': 1 / 3,
    'lgb_test_preds': 1 / 3,
    'xgb_test_preds': 1 / 3,
}

# 根据权重对各元模型的预测结果进行加权平均
cat_test_preds_weighted = submit_pred_2 * weights['cat_test_preds']
lgb_test_preds_weighted = submit_pred_1 * weights['lgb_test_preds']
xgb_test_preds_weighted = submit_pred_3 * weights['xgb_test_preds']
submit_pred = cat_test_preds_weighted + lgb_test_preds_weighted + xgb_test_preds_weighted

# submit_pred = (submit_pred > 0.5).astype(int)

# 创建提交文件，包含测试集ID和预测结果，列名为'label'
submission = pd.DataFrame({
    'user_id': test_user_ids,
    'label': submit_pred  # 最终预测结果列名改为'label'
})

# 保存提交文件到CSV格式
submission.to_csv('../output/submission1.csv', index=False)

# 打印提交文件内容
print(submission)
