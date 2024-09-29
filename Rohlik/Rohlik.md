## 项目技术文档：订单预测挑战

### 项目概述

本项目旨在通过构建多个机器学习模型，对订单数据进行预测。我们通过特征工程、数据预处理、模型训练和预测，结合多种模型（如LightGBM、XGBoost、CatBoost等）及其堆叠集成（Stacking），最终得到订单预测结果，总体模型表现较为突出。

### 数据说明

1.  **训练集 (`train.csv`)**: 包含历史订单数据及目标变量 `orders`。
2.  **测试集 (`test.csv`)**: 不包含目标变量 `orders`，需通过模型进行预测。
3.  **特征列**: 除了 `id` 列，剩余的特征用于训练模型，特征包括 `date`、`warehouse`、`holiday_name` 等。

### 特征工程

特征工程方面，由于该数据集并没有缺失值和异常值，因此在进行特征工程的时候，我针对时间序列构建组合特征，以此提升模型解释性，并将部分特征列转为独热编码。

#### 日期特征处理

*   将日期列转换为多种时间特征，提取的特征如下：

    *   **年份**：`date_year`
    *   **月份**：`date_month`
    *   **日期**：`date_day`
    *   **星期几**：`date_day_of_week`
    *   **一年中的第几周**：`date_week_of_year`
    *   **日期序列数**：距离最早日期的天数 `date_num`
    *   **一年中的第几天**：`date_day_of_year`
    *   **季度**：`date_quarter`
    *   **是否为月初或月末**：`date_is_month_start` 和 `date_is_month_end`
    *   **是否为季度初或季度末**：`date_is_quarter_start` 和 `date_is_quarter_end`

#### 周期性特征

通过对日期特征进行正余弦转换，生成周期性特征：

*   **月份的周期性特征**：`month_sin` 和 `month_cos`
*   **日期的周期性特征**：`day_sin` 和 `day_cos`
*   **年份的周期性特征**：`year_sin` 和 `year_cos`

#### 假期特征处理

*   对缺失的假期名称使用 `'None'` 填充。
*   使用 `OneHotEncoder` 对假期名称进行独热编码。

#### 仓库特征处理

*   使用 `LabelEncoder` 对仓库列进行标签编码，将类别数据转化为整数形式。

#### 假期前后状态

创建了两列新的特征：

*   **假期前**：`holiday_before`，表示当前日期前是否为假期。
*   **假期后**：`holiday_after`，表示当前日期后是否为假期。

### 数据预处理

1.  将训练集和测试集进行合并，以便统一处理特征。
2.  通过日期特征处理和假期特征编码，最终得到带有新特征的数据集。
3.  将数据拆分为训练集（`train_df_le`）和测试集（`test_df_le`）。

### 模型构建

使用了多种回归模型进行预测，并采用了堆叠集成（Stacking）的方式进行模型融合：

#### 基模型

1.  **LightGBM (LGBMRegressor)**
2.  **XGBoost (XGBRegressor)**
3.  **CatBoost (CatBoostRegressor)**
4.  **随机森林 (RandomForestRegressor)**
5.  **AdaBoost (AdaBoostRegressor)**
6.  **决策树 (DecisionTreeRegressor)**
7.  **梯度提升树 (GradientBoostingRegressor)**

每个模型都通过10折交叉验证进行训练，并将验证集上的预测结果保存到堆叠训练矩阵（`stacking_train`）中。

#### 元模型

为了进一步提升预测效果，使用了三种元模型（第二层模型）对堆叠后的基模型输出进行预测：

1.  **LightGBM (LGBMRegressor)**
2.  **CatBoost (CatBoostRegressor)**
3.  **XGBoost (XGBRegressor)**

最终对这些元模型的输出进行加权平均，生成最终预测结果。

### 交叉验证和模型训练

使用 `KFold` 进行10折交叉验证，并且对于每一折交叉验证，训练基模型并在验证集上进行预测。同时，将各个模型在测试集上的预测结果存储下来，以便后续的元模型使用。

### 模型融合

*   使用 `LGBMRegressor`、`CatBoostRegressor` 和 `XGBRegressor` 作为元模型，通过堆叠基模型的预测结果进行训练。
*   使用模型权重对每个元模型的预测结果进行加权平均，最终得到提交结果。

### 模型评价

模型使用两种评价指标：

*   **平均绝对百分比误差 (MAPE)**
*   **R2评分**

### 提交文件生成

最后，将测试集的 `id` 和预测结果 `submit_pred` 保存为 CSV 文件，供提交使用。

### 结果文件

生成的 `submission.csv` 包含两列：

1.  `id`: 测试集的唯一标识。
2.  `Target`: 预测的订单数量。

### 模型权重设置

在最终的预测结果加权中，为三个元模型赋予相等权重：

```
python复制代码weights = {
    'cat_test_preds': 1/3,
    'lgb_test_preds': 1/3,
    'xgb_test_preds': 1/3,
}

```

### 总结

本项目通过多模型融合和堆叠集成，较大程度上提升了订单预测的准确性。使用的特征工程包括丰富的日期特征、类别特征编码、假期处理和周期性特征的提取。模型选择涵盖了多种先进的树模型（如LightGBM、XGBoost、CatBoost等），并通过加权平均的方法融合元模型的结果，最终输出订单预测值。
