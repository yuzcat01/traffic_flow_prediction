# 交通流量预测系统技术设计说明

## 1. 总体架构

项目采用分层设计，核心结构如下：

- `datasets`
  负责数据加载、归一化、切片、建图和预处理。
- `models`
  负责空间模块、时间模块和时空组合模型。
- `trainers`
  负责训练、验证、测试、指标统计和结果落盘。
- `services`
  负责 GUI 所需的业务逻辑，如数据预览、配置管理、模型注册、推理、报表。
- `gui`
  负责页面、样式和用户交互。
- `workers`
  负责 GUI 中的后台训练任务。

## 2. 核心数据流

训练链路如下：

1. 读取 `data/train/model` 三类配置。
2. 构建 `LoadData` 数据集，完成流量预处理、归一化、样本切片和邻接矩阵构建。
3. 通过 `Trainer` 创建 `DataLoader`、模型、优化器和调度器。
4. 执行训练、验证、早停和模型保存。
5. 执行测试并生成指标、曲线、步长指标 JSON 与实验记录。

推理链路如下：

1. 根据运行配置和 checkpoint 重建模型。
2. 复用训练阶段的归一化逻辑和图结构配置。
3. 对窗口输入执行预测并反归一化。
4. 将结果交给 GUI 页面做图表、文本和导出展示。

## 3. 模型设计

### 3.1 空间模块

- `GCNSpatial`
- `ChebNetSpatial`
- `GATSpatial`

三类空间模块输入输出维度已统一，便于公平对比与后续扩展。

### 3.2 时间模块

- `GRUTemporal`
- `IdentityTemporal`

其中 `IdentityTemporal` 主要用于对比实验和无时间模块场景。

### 3.3 组合模型

`STModel` 负责把空间编码与时间编码串联，并通过输出头得到未来多步预测。

当前输出头支持：

- `linear`
- `horizon_mlp`

同时支持残差式“最后时刻值回注”，用于提升预测稳定性。

## 4. 本轮优化点

### 4.1 性能优化

- 将时空模型中的空间编码从“按时间步循环”改为“批量化并行编码”。
- 为 `DataLoader` 增加自定义 `collate_fn`，避免每个样本重复携带同一张图。
- 在训练和推理阶段统一做 batch-to-device，减少模型内部设备迁移逻辑。
- 为 `DataLoader` 补充 `pin_memory` 和 `persistent_workers` 支持，便于 CUDA 环境下进一步提速。

### 4.2 数据处理增强

- 新增 `dataset.preprocess` 配置段。
- 支持缺失值处理：
  - `none`
  - `linear_interpolate`
  - `forward_fill`
  - `mean_fill`
- 支持异常值裁剪：
  - `clip_min`
  - `clip_max_quantile`
- 在数据预览页展示预处理配置和预处理统计信息。

### 4.3 GUI 能力增强

- 数据页新增时空热力图预览。
- 数据配置生成页新增预处理参数入口。
- 数据预览页可同时查看单节点曲线和热力图，更接近开题报告中的展示要求。

## 5. 可扩展设计建议

### 5.1 模型扩展

新增空间模块时，只需：

1. 在 `src/models/spatial/` 中增加实现。
2. 在 `src/models/builder.py` 中注册构造逻辑。
3. 在 `configs/model/` 中提供模板配置。

### 5.2 时间模块扩展

新增时间模块时，只需：

1. 在 `src/models/temporal/` 中增加实现。
2. 让该模块输出统一的 `hidden_dim`。
3. 在 `build_temporal_encoder` 中注册。

### 5.3 GUI 扩展

建议后续将通用画布、页面标题区、指标卡片与表单构造逻辑继续抽象为共享组件，降低页面文件体积。
