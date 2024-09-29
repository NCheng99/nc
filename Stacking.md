# 数据预处理文档

本数据预处理文档详细描述了在分析和建模过程中对训练集和测试集进行的所有数据清理和特征工程步骤。预处理的目的是确保数据的质量，填补缺失值，转换数据类型，并构建新特征，以提升模型的性能和准确性。

## 1.数据准备

```python
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test_b_x.csv')

#提取测试集中的 user_id 列，以便后续可能需要引用
test_user_ids = test_df['user_id']

#将训练集和测试集合并为一个 DataFrame，方便后续处理
all_df = pd.concat([train_df, test_df], sort=False).reset_index(drop=True)

#删除 user_id 列，确保数据集中没有不必要的字段
all_df = all_df.drop(columns=['user_id'], errors='ignore')
```

## 2.数据清洗

```python
#将合约值 -1 替换为 0，表示没有合约,这种处理使得数据更加一致，便于后续分析和模型训练。
all_df['term_cont_plan_months'] = all_df['term_cont_plan_months'].replace(-1, 0)

#将 KB 单位的流量转换为 MB  
all_df['free_app_flux_m3'] = all_df['free_app_flux_m3'] / 1024
```

## 3.缺失值处理

```python
#使用 star_level 分组，填补积分特征列中的缺失值为该组的中位数 按组填补缺失值可以保留数据的分布特性，避免填补均值导致的信息损失
all_df['curmon_acum_qty'] = all_df.groupby('star_level')['curmon_acum_qty'].transform(lambda x: x.fillna(x.median()))

#定义填充函数，填补 curmon_acum_qty 中的缺失值 使用均值填补缺失值有助于保留样本的代表性，同时降低数据噪声
def fill_missing_values(row):
    if pd.isna(row['curmon_acum_qty']):
        return mean_by_star_level.get(row['star_level'], np.nan)
    return row['curmon_acum_qty']

all_df['curmon_acum_qty'] = all_df.apply(fill_missing_values, axis=1)

#在观察缺失值的所在行发现，这些用户可能都是新用户，为了避免出现不必要的噪点，所以考虑用0填补，避免出现负面影响
columns_to_interpolate = ['m1_comm_days', 'm2_comm_days', 'm3_comm_days', 'm1_calling_cnt', 'm2_calling_cnt', 'm3_calling_cnt']
all_df[columns_to_interpolate] = all_df[columns_to_interpolate].fillna(0)

#使用中位数填充该特征列,中位数填充能更好地处理异常值，确保数据的稳健性
columns_to_fill = ['m1_hot_app_flow', 'm2_hot_app_flow', 'm3_hot_app_flow']
all_df[columns_to_fill] = all_df[columns_to_fill].fillna(all_df[columns_to_fill].median())
```

## 4.特征分箱

```python
#定义分箱的函数，对特定列进行分箱操作，生成分箱标签和数值编码,分箱可以将连续变量转化为分类变量，减少模型对噪声的敏感性，同时也能提高模型的解释性。
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
    bins = list(range(0, max_value, bin_interval))
    bins.append(np.inf)
    labels = [f"[{bins[i]}, {bins[i+1]})" for i in range(len(bins) - 1)]
    df[f'binned_{column_name}'] = pd.cut(df[column_name], bins=bins, labels=labels, right=False)
    df[f'binned_{column_name}_label'] = df[f'binned_{column_name}'].cat.codes
    return df

# 对五个特征列，分别进行分箱操作
columns_4000 = ['free_app_flux_m3']
for col in columns_4000:
    all_df = binning_and_labeling(all_df, col, 4000, 40000, bin_label_prefix='flux')

columns_30 = ['domestic_roam_call_count_m3','contact_avg_called_cnt_m3','contact_avg_call_cnt_m3']
for col in columns_30:
    all_df = binning_and_labeling(all_df, col, 30, 300, bin_label_prefix='calling')    
    
columns_20 = ['bw_contact_count_m3',]
for col in columns_20:
    all_df = binning_and_labeling(all_df, col, 20, 200, bin_label_prefix='bw')

# 打印统计结果
for col in columns_4000+columns_30+columns_20:
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
all_df = all_df.drop(columns=[f'binned_{col}' for col in columns_4000+columns_30+columns_20])
```

## 5.特征工程

```python
#手动新增特征以丰富数据集，这些特征可能对模型的预测能力有正面影响 新特征的添加能够引入更多信息，提高模型的表达能力，进而提升预测效果

def add_new_features(df):
    # 手动新增的新特征
    
    #3月联系人被叫通话次数与3月联系人通话次数的比率
    df['contact_avg_called/contact_avg_call'] = (df['contact_avg_called_cnt_m3'] / 
                                                 df['contact_avg_call_cnt_m3'].replace(0, np.nan)).fillna(0)
    # 1-3月缴费金额总和
    df['m1_m2_m3_owe_fee_sum'] = df[['m1_owe_fee', 'm2_owe_fee', 'm3_owe_fee']].sum(axis=1)
    
    # 1-3月缴费金额平均值
    df['m1_m2_m3_owe_fee_average'] = df[['m1_owe_fee', 'm2_owe_fee', 'm3_owe_fee']].mean(axis=1)
    
    # 网龄/年龄比率
    df['innet_months_per_age'] = (df['innet_months'] / df['age'].replace(0, np.nan)).fillna(0)
    
    # 每星级所需网龄均值
    df['innet_months_per_star_level'] = (df['innet_months'] / df['star_level'].replace(0, np.nan)).fillna(0)
    
    # 对新增的特征进行取整，保留小数点后两位
    new_features = [
        'contact_avg_called/contact_avg_call', 'm1_m2_m3_owe_fee_sum', 'm1_m2_m3_owe_fee_average',
        'innet_months_per_age', 'innet_months_per_star_level']
    
    df[new_features] = df[new_features].round(2)

    return df

# 添加新特征
all_df = add_new_features(all_df)

#将新特征中的 NaN 值填充为 0
columns_to_fill_zero = [
    'innet_months_per_age', 
    'innet_months_per_star_level'
]

# 填充新特征中NaN值为0的列
all_df[columns_to_fill_zero] = all_df[columns_to_fill_zero].fillna(0)

#转化数据类型，从而提高模型训练速度
def convert_dtype(df):
    # 将所有 float64 列转换为 float32
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')

    # 将所有 int64 列转换为 int32
    int_cols = df.select_dtypes(include=['int64']).columns
    df[int_cols] = df[int_cols].astype('int32')

    return df
```

# 模型构建文档

## 1. **模型构建思路**

堆叠模型作为一种集成学习方法，通过结合多个基模型的预测结果，旨在提升整体的预测性能。这种方法的核心在于利用不同类型模型的多样性，这些模型在特征空间上具有不同的学习能力和偏差表现，能够在多维度上互补。具体来说，基于不同算法的基模型（LightGBM、XGBoost、CatBoost和随机森林）能够捕捉数据中的不同模式和特征。

![image.png](https://note.youdao.com/yws/res/2/WEBRESOURCEa5e6d4ecb4b383d58d8cba12d4643c82)

·  **互补性**：每种基模型在不同特征或样本分布上表现可能不同，通过组合这些模型，可以最大限度地发挥各个模型的优点，减小单一模型可能带来的偏差。

·  **层次学习**：第一层基模型的预测结果作为第二层模型的输入，使得堆叠模型能够在更高层次上提取信息和特征，进一步优化最终预测。这种层次结构有效增强了模型的表达能力。

## 2. **模型框架**

**2.1基模型（第一层） 在第一层中，选择不同的基模型以确保多样性和鲁棒性:**

·  **LightGBM**：因其高效的训练速度和低内存消耗，特别适合处理大规模和高维稀疏特征的数据。这使得模型能够迅速得到初步的预测结果，并适应大数据场景。

·  **XGBoost**：其强大的正则化机制和对过拟合的控制能力，使其在各类比赛中表现卓越，能够提供稳健的预测。XGBoost的灵活性允许用户根据具体问题进行优化，进一步提升性能。

·  **CatBoost**：专注于类别特征的处理，通过自动化特征编码减少了预处理的复杂性。这使得CatBoost在多种数据类型中表现良好，能够轻松处理复杂特征。

·  **随机森林**：通过构建多个决策树并结合结果，随机森林能够有效降低模型的方差，提高预测的稳健性。它特别适合高维数据，增强了模型的稳定性。

**2.2 二层模型（第二层）**

**在第一层模型的输出基础上，选择二层模型时使用相同的算法（LightGBM、XGBoost、CatBoost）。这种选择的原因主要包括：**

· **特征组合的优势**：使用与第一层相同的模型可以更好地利用其特征组合能力，特别是在处理复杂特征时，二层模型能够捕捉到第一层模型输出之间的潜在关系。

· **适应能力**：二层模型需要对第一层预测结果的适应能力强，能够有效学习如何将这些结果整合成更准确的最终预测。这种融合不仅仅是简单的加权平均，而是通过学习不同模型的输出关系来优化结果。

## **3.思路总结**

我认为通过结合不同类型的基模型，堆叠模型能够在多个方面增强整体性能，包括提升准确性、降低过拟合风险和处理复杂特征。层次结构的设计允许模型在不同层次上进行信息提取和特征组合，从而获得更稳健的预测结果。这样的构造方式不仅增强了模型的表达能力，还提升了其在实际应用中的适应性和可靠性。
