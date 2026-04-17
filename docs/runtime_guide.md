# 本地运行与复现说明

## 1. 推荐环境

项目建议运行在本地 conda 环境中，不建议直接使用系统 Python。

推荐流程：

```bash
conda create -n traffic_gnn python=3.10
conda activate traffic_gnn
pip install -r requirements.txt
```

如果需要 GPU，请安装与本机 CUDA 匹配的 PyTorch。

## 2. 启动方式

### 2.1 启动 GUI

```bash
python run_gui.py
```

Windows 下也可以使用：

```bash
rungui.bat
```

### 2.2 训练单个模型

```bash
python train.py --model_cfg configs/model/gcn_gru.yaml
python train.py --model_cfg configs/model/chebnet_gru.yaml
python train.py --model_cfg configs/model/gat_gru.yaml
```

### 2.3 批量实验

```bash
python run_all.py
```

## 3. 推荐验证顺序

1. 在 conda 环境中执行 `python -c "import torch, yaml, PyQt5"`，确认依赖可用。
2. 启动 GUI，检查首页和数据页是否正常显示。
3. 在数据页加载默认 `PeMS04` 配置，检查节点曲线和热力图是否正常。
4. 在训练页使用默认配置跑通一次实验。
5. 在模型管理页加载结果模型。
6. 在推理页和结果分析页确认指标、曲线和报表导出正常。

## 4. 本轮新增配置说明

`configs/data/*.yaml` 已支持预处理字段：

```yaml
preprocess:
  missing_strategy: linear_interpolate
  clip_min: 0.0
  clip_max_quantile:
```

建议：

- 默认优先尝试 `linear_interpolate`。
- 对噪声较大的自定义数据，可以设置 `clip_max_quantile: 0.995` 或 `0.99` 做上分位裁剪。
- 如果原始数据已经清洗完毕，也可以把 `missing_strategy` 设为 `none`。

## 5. 当前验证说明

截至 2026-04-17，本轮修改除语法级校验外，已经在本地 `traffic_gnn` 环境中完成真实运行验证。

已实际完成的运行包括：

- `chebnet_gru_h3` 训练与测试
- `gcn_gru_h3` 训练与测试
- `gat_gru_h3` 训练与测试
- `tests.test_model_blocks` 单元测试

如果需要查看这轮真实实验的详细指标、分步长结果和结论建议，请继续阅读 `docs/experiment_results.md`。
