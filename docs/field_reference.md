# 配置字段说明

## 1. 数据配置 `configs/data/*.yaml`

示例：

```yaml
dataset:
  name: PeMS04
  graph_path: data/raw/PeMS04/PeMS04.csv
  flow_path: data/raw/PeMS04/PeMS04.npz
  num_nodes: 307
  divide_days: [45, 14]
  time_interval: 5
  preprocess:
    missing_strategy: linear_interpolate
    clip_min: 0.0
    clip_max_quantile:
```

字段说明：

- `name`
  数据集名称。
- `graph_path`
  图结构文件路径，通常为 `csv` 或 `txt`。
- `flow_path`
  流量数据路径，通常为 `npz`。
- `num_nodes`
  节点数量。
- `divide_days`
  训练天数和测试天数，格式为 `[train_days, test_days]`。
- `time_interval`
  采样间隔，单位为分钟。
- `preprocess.missing_strategy`
  缺失值处理策略，可选 `none`、`linear_interpolate`、`forward_fill`、`mean_fill`。
- `preprocess.clip_min`
  最小裁剪值，小于该值的数据会被裁剪。
- `preprocess.clip_max_quantile`
  上分位裁剪阈值，取值范围 `(0, 1)`，留空表示不启用。

## 2. 训练配置 `configs/train/*.yaml`

常见字段：

- `epochs`
  训练轮数。
- `batch_size`
  批大小。
- `learning_rate`
  学习率。
- `weight_decay`
  权重衰减。
- `optimizer`
  优化器，当前支持 `adam`、`adamw`。
- `grad_clip_norm`
  梯度裁剪阈值，`0` 表示关闭。
- `lr_scheduler`
  学习率调度器，当前支持 `plateau`、`none`。
- `lr_scheduler_factor`
  调度器衰减系数。
- `lr_scheduler_patience`
  调度器等待轮数。
- `min_lr`
  最小学习率。
- `seed`
  随机种子。
- `num_workers`
  `DataLoader` 工作线程数。
- `shuffle`
  训练集是否打乱。
- `device`
  设备配置，建议使用 `auto`、`cpu` 或 `cuda`。
- `val_ratio`
  验证集比例。
- `early_stop_patience`
  早停等待轮数。
- `early_stop_min_delta`
  早停最小改进量。
- `save_dir`
  结果输出目录。
- `figure_node_id`
  预测曲线绘制的节点编号。
- `figure_points`
  绘制曲线的采样点数。
- `figure_horizon_step`
  绘制第几个预测步的曲线，索引从 `0` 开始。
- `loss_fn`
  损失函数，支持 `mse`、`mae`、`huber`。
- `huber_delta`
  Huber 损失参数。
- `horizon_weight_mode`
  多步预测步长加权模式，支持 `uniform`、`linear_decay`、`exp_decay`、`custom`。
- `horizon_weight_gamma`
  指数衰减权重参数。
- `horizon_weights`
  自定义步长权重列表，仅在 `custom` 下使用。

## 3. 模型配置 `configs/model/*.yaml`

示例：

```yaml
model:
  name: gcn_gru
  graph:
    type: connect
  input:
    history_length: 12
    input_dim: 1
  spatial:
    type: gcn
    hidden_dim: 16
  temporal:
    type: gru
    hidden_dim: 32
    num_layers: 1
  regularization:
    dropout: 0.10
  output:
    output_dim: 1
    predict_steps: 1
    head_type: horizon_mlp
    pred_hidden_dim: 64
    horizon_emb_dim: 8
    dropout: 0.10
    use_last_value_residual: true
```

字段说明：

- `model.name`
  实验名或模型名，会参与结果文件命名。
- `graph.type`
  建图方式，支持 `connect`、`distance`、`correlation`、`distance_correlation`。
- `graph.correlation_topk`
  相关图保留的每行 Top-K 邻居数。
- `graph.correlation_threshold`
  相关图阈值。
- `graph.use_abs_corr`
  是否对相关系数取绝对值。
- `graph.fusion_alpha`
  距离图与相关图的融合权重。
- `input.history_length`
  输入历史窗口长度。
- `input.input_dim`
  输入特征维度，当前默认 `1`。
- `spatial.type`
  空间模块类型，支持 `gcn`、`chebnet`、`gat`。
- `spatial.hidden_dim`
  空间隐藏维度。
- `spatial.cheb_k`
  ChebNet 的阶数参数。
- `spatial.heads`
  GAT 注意力头数。
- `temporal.type`
  时间模块类型，支持 `gru`、`tcn`、`none`。
- `temporal.hidden_dim`
  时间模块隐藏维度。
- `temporal.num_layers`
  时间模块层数，适用于 `GRU` 和 `TCN`。
- `temporal.kernel_size`
  `TCN` 卷积核大小，默认 `3`。
- `regularization.dropout`
  通用 dropout 比例。
- `output.output_dim`
  输出维度。
- `output.predict_steps`
  预测步数。
- `output.head_type`
  输出头类型，支持 `linear`、`horizon_mlp`。
- `output.pred_hidden_dim`
  `horizon_mlp` 隐藏层维度。
- `output.horizon_emb_dim`
  步长嵌入维度。
- `output.dropout`
  输出头 dropout。
- `output.use_last_value_residual`
  是否使用最后观测值残差回注。
