# Traffic Flow Prediction

基于图神经网络的交通流量预测系统，面向毕业设计课题“基于图神经网络的交通流量预测系统的设计与实现”。

当前项目已经覆盖这些主线能力：

- `GCN`、`ChebNet`、`GAT` 三类空间建模模块
- `GRU` / `none` 两类时间建模模块
- 训练、测试、结果记录、预测推理
- PyQt GUI：数据预览、训练管理、模型管理、结果分析、推理展示
- 多种建图策略：`connect`、`distance`、`correlation`、`distance_correlation`

## 目录概览

- `configs/`: 数据、训练、模型配置
- `datasets/`: 数据加载与图构建
- `models/`: 时空模型实现
- `services/`: GUI 与业务服务层
- `trainers/`: 训练与测试流程
- `gui/`: 可视化界面
- `results/`: 实验结果、图像、checkpoint、运行配置

## 快速开始

安装依赖：

```bash
pip install -r requirements.txt
```

命令行训练：

```bash
python train.py --model_cfg configs/model/gcn_gru.yaml
python train.py --model_cfg configs/model/gcn_gru_correlation.yaml
python train.py --model_cfg configs/model/gcn_gru_distance_correlation.yaml
```

启动 GUI：

```bash
python run_gui.py
```

GUI 固定图标文件：

- `gui/assets/app_icon.ico`（推荐用于打包）
- `gui/assets/app_icon.png`

PyInstaller 打包（Windows）：

推荐使用 `onedir`（目录模式），对包含 `torch` 的项目更稳定、启动更快、排错更容易。

一键脚本（推荐）：

```bash
build_exe.bat
```

说明：不带参数时默认 `onedir + lite`（仅 GUI 展示，不含 torch）。

脚本可选模式：

```bash
build_exe.bat onedir
build_exe.bat onefile
build_exe.bat onedir lite
build_exe.bat onefile full
build_exe.bat onedir lite nopause
```

手动命令示例：

```bash
pyinstaller -F -w run_gui.py --icon gui/assets/app_icon.ico --add-data "gui/assets;gui/assets"
```

如果希望可执行文件支持训练/推理（包含 `torch`）：

```bash
pyinstaller -F -w run_gui.py --icon gui/assets/app_icon.ico --add-data "gui/assets;gui/assets" --collect-all torch
```

说明：

- 未打入 `torch` 时，程序仍可启动和浏览部分页面，但训练/推理会提示缺少依赖。
- `onefile` 首次启动会有解包开销，体积更大；`onedir` 一般体验更好。
- `requirements.txt` 用于源码环境安装，不会自动作用于已打包的 `exe`。
- `exe` 打包完成后，后续再安装 `torch` 不会“修复旧 exe”，需要重新打包。

批量实验：

```bash
python run_all.py
```

## 配置文件（YAML）编写与生成

你可以用两种方式准备配置文件：

- 在 GUI 的“实验训练”页面，点击“生成默认训练配置”“生成默认模型配置”
- 手动在 `configs/data`、`configs/train`、`configs/model` 下新建 `.yaml`

默认生成文件：

- `configs/train/default_generated.yaml`
- `configs/model/gcn_gru_generated.yaml`
- `configs/model/gcn_gru_corr_generated.yaml`

### 1) 数据配置 `configs/data/*.yaml`

```yaml
dataset:
  name: PeMS04
  graph_path: data/raw/PeMS04/PeMS04.csv
  flow_path: data/raw/PeMS04/PeMS04.npz
  num_nodes: 307
  divide_days: [45, 14]
  time_interval: 5
```

字段说明：

- `graph_path`: 边关系文件（csv）
- `flow_path`: 流量时间序列文件（npz）
- `divide_days`: `[训练天数, 测试天数]`
- `time_interval`: 采样间隔（分钟）

### 2) 训练配置 `configs/train/*.yaml`

```yaml
train:
  epochs: 50
  batch_size: 64
  learning_rate: 0.001
  weight_decay: 0.0001
  optimizer: adamw
  grad_clip_norm: 5.0
  lr_scheduler: plateau
  lr_scheduler_factor: 0.5
  lr_scheduler_patience: 3
  min_lr: 0.00001
  seed: 42
  num_workers: 0
  shuffle: true
  device: auto
  val_ratio: 0.1
  early_stop_patience: 8
  early_stop_min_delta: 0.0001
  save_dir: results
  figure_node_id: 0
  figure_points: 300
  figure_horizon_step: 0
  loss_fn: mse
  huber_delta: 1.0
  horizon_weight_mode: uniform
  horizon_weight_gamma: 0.9
  horizon_weights: []
```

字段说明：

- `loss_fn`: `mse` / `mae` / `huber`
- `horizon_weight_mode`: `uniform` / `linear_decay` / `exp_decay` / `custom`
- 当 `horizon_weight_mode=custom` 时，`horizon_weights` 长度必须等于 `predict_steps`

### 3) 模型配置 `configs/model/*.yaml`

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

常用可选值：

- `spatial.type`: `gcn` / `chebnet` / `gat`
- `temporal.type`: `gru` / `none`
- `graph.type`: `connect` / `distance` / `correlation` / `distance_correlation`

## 建图策略

模型配置中的 `model.graph` 现在支持以下策略：

- `connect`: 读取边文件，存在连接则置为 `1`
- `distance`: 使用边文件中的距离倒数作为权重
- `correlation`: 基于训练阶段流量序列的节点相关性建图
- `distance_correlation`: 将距离图与相关性图按比例融合

相关参数示例：

```yaml
model:
  graph:
    type: distance_correlation
    correlation_topk: 8
    correlation_threshold: 0.30
    fusion_alpha: 0.5
```

字段说明：

- `correlation_topk`: 每个节点保留的 strongest 邻居数
- `correlation_threshold`: 低于该阈值的相关性边会被裁剪
- `use_abs_corr`: 是否使用相关系数绝对值
- `fusion_alpha`: 混合图中距离图的权重

## 实验输出

训练完成后会自动生成：

- `results/checkpoints/*.pth`
- `results/figures/*_loss_curve.png`
- `results/figures/*_prediction.png`
- `results/run_configs/*.json`
- `results/metrics_summary.csv`

这几类结果可以直接用于论文中的实验复现、对比分析和 GUI 展示。
