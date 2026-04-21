# Traffic Flow Prediction

基于图神经网络的交通流量预测系统，面向“基于图神经网络的交通流量预测系统的设计与实现”相关课程设计与毕业设计场景。项目同时支持命令行训练、批量实验、多模型结果分析，以及基于 PyQt 的图形界面操作。

当前项目已经覆盖这些核心能力：

- 支持 `GCN`、`ChebNet`、`GAT` 三类空间建模模块
- 支持 `GRU` / `TCN` / `none` 三类时间建模模块
- 支持缺失值处理与基础异常值裁剪预处理
- 支持训练、测试、结果记录、模型推理
- 支持多种建图策略：`connect`、`distance`、`correlation`、`distance_correlation`
- 提供 PyQt GUI：数据预览、实验训练、模型管理、在线推理、结果分析
- 提供应用决策原型：基于预测流量进行拥堵预警与节点级路线推荐
- 提供节点曲线、指标图表、时空热力图、报告导出等分析能力

## 项目结构

```text
traffic_flow_prediction/
├─ src/                              # 核心源码
│  ├─ datasets/                      # 数据读取与图构建
│  ├─ models/                        # 时空模型定义
│  ├─ trainers/                      # 训练与测试流程
│  ├─ services/                      # GUI 业务服务
│  ├─ gui/                           # 界面与页面组件
│  ├─ workers/                       # GUI 后台任务
│  ├─ utils/                         # 通用工具
│  └─ project_paths.py               # 项目路径解析
├─ configs/                          # 数据 / 训练 / 模型配置
│  ├─ data/
│  ├─ train/
│  └─ model/
├─ data/                             # 原始数据与导入数据
├─ results/                          # 实验结果、图像、报告、checkpoint
├─ run_gui.py                        # GUI 入口
├─ train.py                          # 单模型训练入口
├─ run_all.py                        # 批量实验入口
├─ rungui.bat                        # Windows 双击启动 GUI
├─ build_exe.bat                     # Windows 打包脚本
├─ requirements.txt                  # 运行依赖
└─ README.md
```

## 文档入口

完整文档已经整理到 `docs/` 目录，建议优先从下面几个入口阅读：

- `docs/README.md`：文档索引总入口
- `docs/requirements.md`：需求说明
- `docs/technical_design.md`：技术设计说明
- `docs/application_decision.md`：拥堵预警与路线推荐应用模块说明
- `docs/field_reference.md`：配置字段说明
- `docs/runtime_guide.md`：本地运行与复现说明
- `docs/compliance_and_sustainability.md`：数据合规、隐私边界与资源消耗说明
- `docs/experiment_results.md`：`h3 / h6 / h12` 三组真实实验结果
- `docs/paper_tables.md`：论文表格与正文引用模板
- `docs/defense_summary.md`：答辩讲解提纲
- `docs/ppt_outline.md`：详细版 PPT 文案提纲
- `docs/ppt_brief.md`：5 分钟左右精简答辩稿
- `docs/defense_qa.md`：答辩常见问题清单

## 快速开始

安装依赖：

```bash
pip install -r requirements.txt
```

说明：

- `requirements.txt` 主要覆盖源码运行与 GUI 使用所需依赖
- 如果需要实际训练和推理，请确认当前环境中已经安装与你设备匹配的 `torch`

启动 GUI：

```bash
python run_gui.py
```

Windows 下也可以直接双击：

```bash
rungui.bat
```

命令行训练单个模型：

```bash
python train.py --model_cfg configs/model/gcn_gru.yaml
python train.py --model_cfg configs/model/gcn_gru_correlation.yaml
python train.py --model_cfg configs/model/gcn_gru_distance_correlation.yaml
```

批量实验：

```bash
python run_all.py
```

## 环境配置建议

推荐优先使用独立虚拟环境或 conda 环境，避免和本机已有的 Python 包冲突。

### 方式 1：使用 conda

```bash
conda create -n traffic_gnn python=3.10
conda activate traffic_gnn
pip install -r requirements.txt
```

如果你需要训练或推理，请额外安装与你设备匹配的 PyTorch。示例：

```bash
pip install torch torchvision torchaudio
```

如果你使用的是 CUDA 版本的 PyTorch，请以 PyTorch 官方安装说明为准。

项目当前实际验证使用的环境名为：

```bash
traffic_gnn
```

### 方式 2：使用 venv

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

### 设备配置建议

- 如果你有可用 GPU，建议在 `configs/train/default.yaml` 中使用 `device: cuda`
- 如果你只在 CPU 上运行，建议改为 `device: cpu`
- 如果你希望自动判断设备，可以改为 `device: auto`

## 数据准备

当前仓库已经包含默认数据集 `PeMS04`，默认配置文件为：

```text
configs/data/pems04.yaml
```

对应的默认数据路径为：

```text
data/raw/PeMS04/PeMS04.csv
data/raw/PeMS04/PeMS04.npz
```

如果你直接使用仓库内已有数据，通常不需要额外准备，安装依赖后即可运行训练或 GUI。

### 使用自定义数据

你可以通过两种方式接入自己的数据：

- 在 GUI 的“数据管理”页面中导入图结构文件和流量文件，并生成新的数据配置
- 手动把数据放入 `data/` 目录，然后在 `configs/data/*.yaml` 中新增配置文件

建议的数据格式：

- 图结构文件：`.csv` 或 `.txt`
- 流量文件：`.npz`

导入后的数据通常会出现在：

```text
data/raw/imported/
```

对应的数据配置可以保存在：

```text
configs/data/imported_dataset.yaml
```

## 推荐使用流程

对于第一次接触这个项目的同学，建议按下面的顺序使用：

1. 安装依赖，并确认 `torch` 可正常导入
2. 先运行 `python run_gui.py`，确认图形界面可以正常启动
3. 在“数据管理”页面检查默认数据集是否能够正常预览
4. 在“实验训练”页面先选择默认配置，跑通一次基础训练
5. 在“模型管理”页面加载训练完成的模型
6. 在“在线推理”页面查看预测效果
7. 在“应用决策”页面选择起终点和推荐策略，查看拥堵预警与路线建议
8. 在“结果分析”页面对比不同模型、不同建图方式的结果

如果你只想快速验证代码流程，也可以直接先运行：

```bash
python train.py --model_cfg configs/model/gcn_gru.yaml
```

## 应用决策功能

项目新增“应用决策”页面，用于展示交通流量预测模型在真实业务中的使用方式。该页面复用当前加载模型的在线推理结果，将未来节点流量转换为拥堵指数，并结合 `PeMS04.csv` 中的路网边距离生成路线建议。

当前支持：

- 历史测试样本回放，模拟实时路网流量输入
- 起点节点、终点节点、预测步长和推荐策略选择，并提供“选择最远预测步长”快捷按钮
- 三类推荐策略：距离最短、优先避堵、综合最优
- 根据路网连接关系绘制整体拓扑预览
- 支持多条候选路线对比，并在路网图中高亮当前选中路线
- 全网未来拥堵风险 Top-K 节点识别
- 推荐路线节点级预测流量、参考流量和拥堵等级展示
- 路线推荐报告 CSV 导出

说明：PeMS04 默认数据集提供的是传感器节点拓扑和边距离，不包含真实经纬度、道路名称、限速或车道数。因此该功能定位为“基于传感器节点拓扑的路线推荐原型”，适合用于课程设计、论文和答辩中展示预测模型如何进入交通辅助决策环节。

仓库还提供了一个轻量模拟数据集用于验证通用性：

```text
data/raw/sim_demo/sim_demo.csv
data/raw/sim_demo/sim_demo.npz
configs/data/sim_demo.yaml
```

该数据集包含 24 个模拟路网节点、43 条连接边和 10 天 15 分钟粒度的流量序列。训练或预览时可以通过 `--data_cfg configs/data/sim_demo.yaml` 使用它，例如：

```bash
python train.py --data_cfg configs/data/sim_demo.yaml --model_cfg configs/model/gcn_gru_h3.yaml
```

## 配置文件说明

项目中的配置文件放在 `configs/` 目录下，分为三类：

- `configs/data/*.yaml`：数据集路径、节点数、训练/测试天数、时间间隔
- `configs/train/*.yaml`：训练超参数、设备、早停、损失函数、可视化参数
- `configs/model/*.yaml`：图构建方式、空间模块、时间模块、输出头结构

当前仓库内常用配置包括：

- 数据配置：`configs/data/pems04.yaml`
- 训练配置：`configs/train/default.yaml`、`configs/train/gat_safe.yaml`
- 模型配置：
  - `configs/model/gcn_gru.yaml`
  - `configs/model/gcn_gru_correlation.yaml`
  - `configs/model/gcn_gru_distance_correlation.yaml`
  - `configs/model/gcn_none.yaml`
  - `configs/model/chebnet_gru.yaml`
  - `configs/model/chebnet_none.yaml`
  - `configs/model/gat_gru.yaml`
  - `configs/model/gat_none.yaml`

### 1) 数据配置示例

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

- `graph_path`：图结构文件路径，通常为 `.csv`
- `flow_path`：流量序列文件路径，通常为 `.npz`
- `num_nodes`：节点数量
- `divide_days`：`[训练天数, 测试天数]`
- `time_interval`：采样间隔，单位为分钟

### 2) 训练配置示例

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

字段补充：

- `loss_fn`：支持 `mse` / `mae` / `huber`
- `horizon_weight_mode`：支持 `uniform` / `linear_decay` / `exp_decay` / `custom`
- 当 `horizon_weight_mode=custom` 时，`horizon_weights` 长度必须等于 `predict_steps`

### 3) 模型配置示例

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

常见可选值：

- `spatial.type`：`gcn` / `chebnet` / `gat`
- `temporal.type`：`gru` / `tcn` / `none`
- `graph.type`：`connect` / `distance` / `correlation` / `distance_correlation`

## 建图策略

`model.graph` 当前支持以下几种策略：

- `connect`：根据边文件是否连接构造邻接矩阵
- `distance`：使用距离倒数作为边权重
- `correlation`：根据训练阶段流量序列的相关性建图
- `distance_correlation`：将距离图和相关性图按比例融合

相关配置示例：

```yaml
model:
  graph:
    type: distance_correlation
    correlation_topk: 8
    correlation_threshold: 0.30
    fusion_alpha: 0.5
```

字段说明：

- `correlation_topk`：每个节点保留的相关邻居数
- `correlation_threshold`：低于该阈值的相关边会被裁剪
- `use_abs_corr`：是否使用相关系数绝对值
- `fusion_alpha`：融合图中距离图的权重

## GUI 功能概览

图形界面主要包含以下页面：

- 数据管理：数据导入、配置生成、图结构统计、节点曲线预览
- 实验训练：选择配置、覆盖训练参数、启动或停止训练
- 模型管理：筛选实验记录、查看完整配置、加载指定模型
- 在线推理：查看样本预测结果、误差分布、导出推理结果
- 结果分析：查看当前模型指标、分组排名、多模型对比和报告导出

## 典型实验命令

下面给出一些常用实验命令，方便直接复现和对比。

### 1) 训练基础 GCN + GRU 模型

```bash
python train.py --model_cfg configs/model/gcn_gru.yaml
```

### 2) 训练相关性建图版本

```bash
python train.py --model_cfg configs/model/gcn_gru_correlation.yaml
```

### 3) 训练距离图 + 相关图融合版本

```bash
python train.py --model_cfg configs/model/gcn_gru_distance_correlation.yaml
```

### 4) 使用自定义训练配置

```bash
python train.py --train_cfg configs/train/default.yaml --model_cfg configs/model/gat_gru.yaml
```

### 5) 批量运行多个模型

```bash
python run_all.py
```

### 6) 批量运行指定模型与多个随机种子

```bash
python run_all.py --only_models gcn_gru.yaml,gat_gru.yaml --seeds 42,2026,3407
```

### 7) 只查看将要执行的批量任务

```bash
python run_all.py --dry_run
```

### 8) 运行轻量单元测试

```bash
python -m unittest tests.test_model_blocks
```

## 实验输出

训练完成后，项目会自动在 `results/` 下生成结果文件，例如：

- `results/checkpoints/*.pth`
- `results/figures/*_loss_curve.png`
- `results/figures/*_prediction_overview.png`
- `results/figures/*_node*_h*_prediction.png`
- `results/horizon_metrics/*.json`
- `results/run_configs/*.json`
- `results/metrics_summary.csv`
- `results/baseline_runs.csv`
- `results/baseline_summary.csv`
- `results/reports/`

这些输出可直接用于实验对比、图表展示、结果复现和 GUI 页面加载。

## 打包说明

Windows 下推荐使用仓库内的打包脚本：

```bash
build_exe.bat
```

默认行为：

- `build_exe.bat`：默认 `onedir + lite`
- `lite`：仅保证 GUI 本体运行，不主动打入 `torch`
- `full`：打包时额外收集 `torch`，适合希望 exe 直接支持训练 / 推理的场景

常见命令：

```bash
build_exe.bat onedir
build_exe.bat onefile
build_exe.bat onedir lite
build_exe.bat onefile full
build_exe.bat onedir lite nopause
```

说明：

- `onedir` 通常比 `onefile` 更稳定，也更便于排查问题
- 如果当前 Python 环境没有安装 `torch`，则无法执行 `full` 打包
- 已经打包好的 exe 不会因为你之后又安装了依赖而自动修复，需要重新打包

## 开发约定

当前项目已经统一为以 `src/` 为核心源码目录的结构，开发时建议遵循以下约定：

- 统一使用 `src.*` 包导入
- 配置、数据、结果路径统一按项目根目录解析
- 根目录只保留真正对外的入口脚本
- 核心实现尽量集中在 `src/` 内，避免再把业务逻辑散落到根目录

## 提交前自检

如果你准备做最终演示、提交代码或继续扩展，建议至少完成下面几项检查：

1. 在目标环境中确认依赖可正常导入：

```bash
python -c "import torch, yaml, PyQt5"
```

2. 运行轻量单元测试：

```bash
python -m unittest tests.test_model_blocks
```

3. 至少跑通一个单模型训练：

```bash
python train.py --train_cfg configs/train/multistep.yaml --model_cfg configs/model/chebnet_gru_h3.yaml
```

4. 打开 GUI 检查数据页、模型管理页和结果页是否能正常加载：

```bash
python run_gui.py
```

5. 如果需要准备论文或答辩材料，优先查看：

```text
docs/experiment_results.md
docs/paper_tables.md
docs/defense_summary.md
docs/ppt_brief.md
docs/defense_qa.md
```

## 常见问题

### 1) `No module named 'torch'`

说明当前环境没有安装 PyTorch。GUI 的部分页面可能还能打开，但训练、推理、模型加载都会失败。

处理方式：

- 安装与你当前环境匹配的 `torch`
- 如果你在打包 exe，请使用包含 `torch` 的 `full` 模式重新打包

### 2) GUI 可以打开，但加载模型或推理失败

常见原因：

- 当前环境缺少 `torch`
- `results/` 下对应的 checkpoint 文件不存在
- 历史实验记录中的路径已经失效

建议先检查：

- `results/checkpoints/` 中是否存在对应 `.pth`
- `results/run_configs/` 中是否存在对应运行配置
- 当前 Python 环境中是否能正常 `import torch`

### 3) 训练很慢或无法使用 GPU

请检查：

- `configs/train/default.yaml` 中的 `device` 是否设置为 `cuda`
- 当前环境安装的 PyTorch 是否支持 CUDA
- 使用 `python -c "import torch; print(torch.cuda.is_available())"` 检查 GPU 是否可用

如果没有 GPU，可以把 `device` 改为 `cpu` 或 `auto`。

### 4) 批量实验运行后没有看到汇总结果

请检查以下文件是否生成：

- `results/baseline_runs.csv`
- `results/baseline_summary.csv`
- `results/metrics_summary.csv`

如果中途某些实验失败，`run_all.py` 仍可能提前结束或只生成部分结果。

### 5) 打包后的 exe 缺少训练 / 推理能力

这通常是因为你用了 `lite` 模式，或者打包时当前环境没有 `torch`。

建议使用：

```bash
build_exe.bat onefile full
```

或者：

```bash
build_exe.bat onedir full
```

### 6) 从不同目录启动时找不到配置或数据

当前项目已经统一按项目根目录解析路径。正常情况下，直接在项目根目录中运行以下入口即可：

```bash
python run_gui.py
python train.py --model_cfg configs/model/gcn_gru.yaml
python run_all.py
```
