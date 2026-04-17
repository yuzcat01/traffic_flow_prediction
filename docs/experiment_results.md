# 实验结果与结论说明

## 1. 实验目的

本文档用于记录当前项目在本地 `conda` 环境中的真实对照实验结果，重点回答以下问题：

- 当前 `GCN / ChebNet / GAT` 三类空间模块在相同多步预测任务下的效果差异。
- 在精度、训练速度、测试速度之间，哪一种模型更适合作为当前项目的主推方案。
- 后续继续优化时，应该优先投入在哪一类模型上。

## 2. 实验环境与配置

实验执行日期：

- 2026-04-17

实验环境：

- `conda` 环境名：`traffic_gnn`
- 设备：`cuda`

当前文档已记录两组多步预测设置：

- `h3`，即预测未来 `3` 个时间步
- `h6`，即预测未来 `6` 个时间步
- `h12`，即预测未来 `12` 个时间步

使用命令如下：

```bash
conda run -n traffic_gnn python train.py --train_cfg configs/train/multistep.yaml --model_cfg configs/model/chebnet_gru_h3.yaml
conda run -n traffic_gnn python train.py --train_cfg configs/train/multistep.yaml --model_cfg configs/model/gcn_gru_h3.yaml
conda run -n traffic_gnn python train.py --train_cfg configs/train/gat_multistep_safe.yaml --model_cfg configs/model/gat_gru_h3.yaml
conda run -n traffic_gnn python train.py --train_cfg configs/train/multistep.yaml --model_cfg configs/model/chebnet_gru_h6.yaml
conda run -n traffic_gnn python train.py --train_cfg configs/train/multistep.yaml --model_cfg configs/model/gcn_gru_h6.yaml
conda run -n traffic_gnn python train.py --train_cfg configs/train/gat_multistep_safe.yaml --model_cfg configs/model/gat_gru_h6.yaml
conda run -n traffic_gnn python train.py --train_cfg configs/train/multistep.yaml --model_cfg configs/model/chebnet_gru_h12.yaml
conda run -n traffic_gnn python train.py --train_cfg configs/train/multistep.yaml --model_cfg configs/model/gcn_gru_h12.yaml
conda run -n traffic_gnn python train.py --train_cfg configs/train/gat_multistep_safe.yaml --model_cfg configs/model/gat_gru_h12.yaml
```

## 3. 总体结果对比

### 3.0 `h3` 对比结果

### 3.1 总体指标

| 模型 | MAE | MAPE(%) | RMSE |
|---|---:|---:|---:|
| `chebnet_gru_h3` | 19.9822 | 11.9502 | 31.3444 |
| `gcn_gru_h3` | 21.8518 | 12.7746 | 34.4946 |
| `gat_gru_h3` | 19.9459 | 12.0305 | 31.2674 |

### 3.2 训练与测试速度

| 模型 | 训练总耗时(s) | 测试耗时(s) | 测试吞吐(samples/s) |
|---|---:|---:|---:|
| `chebnet_gru_h3` | 101.64 | 1.75 | 2307.2 |
| `gcn_gru_h3` | 160.10 | 2.05 | 1963.9 |
| `gat_gru_h3` | 533.71 | 7.81 | 515.8 |

### 3.3 分步长指标

`chebnet_gru_h3`

| 步长 | MAE | MAPE(%) | RMSE |
|---|---:|---:|---:|
| `H1` | 18.3031 | 11.0066 | 28.9408 |
| `H2` | 20.0348 | 11.9310 | 31.3789 |
| `H3` | 21.6087 | 12.9129 | 33.5440 |

`gcn_gru_h3`

| 步长 | MAE | MAPE(%) | RMSE |
|---|---:|---:|---:|
| `H1` | 20.3698 | 11.8923 | 32.3971 |
| `H2` | 21.8502 | 12.7416 | 34.5186 |
| `H3` | 23.3354 | 13.6899 | 36.4490 |

`gat_gru_h3`

| 步长 | MAE | MAPE(%) | RMSE |
|---|---:|---:|---:|
| `H1` | 18.3247 | 10.9767 | 28.9785 |
| `H2` | 19.9867 | 12.0116 | 31.3162 |
| `H3` | 21.5264 | 13.1033 | 33.3542 |

### 3.4 `h6` 总体指标

| 模型 | MAE | MAPE(%) | RMSE |
|---|---:|---:|---:|
| `chebnet_gru_h6` | 21.6788 | 13.4999 | 33.5624 |
| `gcn_gru_h6` | 23.6969 | 13.9490 | 37.1094 |
| `gat_gru_h6` | 21.2719 | 12.5492 | 33.3330 |

### 3.5 `h6` 训练与测试速度

| 模型 | 训练总耗时(s) | 测试耗时(s) | 测试吞吐(samples/s) |
|---|---:|---:|---:|
| `chebnet_gru_h6` | 289.97 | 4.70 | 857.1 |
| `gcn_gru_h6` | 300.51 | 2.57 | 1566.7 |
| `gat_gru_h6` | 883.50 | 7.63 | 527.8 |

### 3.6 `h6` 分步长指标

`chebnet_gru_h6`

| 步长 | MAE | MAPE(%) | RMSE |
|---|---:|---:|---:|
| `H1` | 18.1716 | 11.0626 | 28.7316 |
| `H2` | 19.8823 | 12.4362 | 31.0080 |
| `H3` | 21.3381 | 13.3500 | 32.9591 |
| `H4` | 22.4713 | 14.0123 | 34.5086 |
| `H5` | 23.5586 | 14.7637 | 35.9594 |
| `H6` | 24.6507 | 15.3742 | 37.4351 |

`gcn_gru_h6`

| 步长 | MAE | MAPE(%) | RMSE |
|---|---:|---:|---:|
| `H1` | 20.4737 | 12.0680 | 32.5818 |
| `H2` | 21.8868 | 12.8783 | 34.6319 |
| `H3` | 23.2345 | 13.5724 | 36.4535 |
| `H4` | 24.4587 | 14.4019 | 38.0409 |
| `H5` | 25.5205 | 14.9862 | 39.4543 |
| `H6` | 26.6073 | 15.7870 | 40.8545 |

`gat_gru_h6`

| 步长 | MAE | MAPE(%) | RMSE |
|---|---:|---:|---:|
| `H1` | 18.0428 | 10.7006 | 28.6831 |
| `H2` | 19.5879 | 11.5351 | 30.9037 |
| `H3` | 20.8780 | 12.3313 | 32.6820 |
| `H4` | 21.9817 | 13.0029 | 34.2034 |
| `H5` | 23.0359 | 13.5753 | 35.6583 |
| `H6` | 24.1049 | 14.1498 | 37.1388 |

### 3.7 `h12` 总体指标

| 模型 | MAE | MAPE(%) | RMSE |
|---|---:|---:|---:|
| `chebnet_gru_h12` | 24.8547 | 14.7903 | 38.2440 |
| `gcn_gru_h12` | 27.2826 | 16.4128 | 42.0460 |
| `gat_gru_h12` | 24.8498 | 14.6548 | 38.3708 |

### 3.8 `h12` 训练与测试速度

| 模型 | 训练总耗时(s) | 测试耗时(s) | 测试吞吐(samples/s) |
|---|---:|---:|---:|
| `chebnet_gru_h12` | 224.40 | 3.37 | 1191.7 |
| `gcn_gru_h12` | 214.71 | 4.43 | 906.9 |
| `gat_gru_h12` | 797.30 | 10.25 | 392.3 |

### 3.9 `h12` 分步长指标

`chebnet_gru_h12`

| 步长 | MAE | MAPE(%) | RMSE |
|---|---:|---:|---:|
| `H1` | 18.0878 | 10.8225 | 28.6563 |
| `H2` | 19.6833 | 11.7242 | 30.9033 |
| `H3` | 21.1164 | 12.6420 | 32.8438 |
| `H4` | 22.2554 | 13.2826 | 34.3898 |
| `H5` | 23.2787 | 13.8912 | 35.8212 |
| `H6` | 24.2878 | 14.4227 | 37.1971 |
| `H7` | 25.3985 | 15.0794 | 38.6936 |
| `H8` | 26.5071 | 15.6917 | 40.1152 |
| `H9` | 27.5923 | 16.4106 | 41.5729 |
| `H10` | 28.7637 | 17.0083 | 43.1403 |
| `H11` | 29.9024 | 17.7091 | 44.6207 |
| `H12` | 31.3830 | 18.7983 | 46.4771 |

`gcn_gru_h12`

| 步长 | MAE | MAPE(%) | RMSE |
|---|---:|---:|---:|
| `H1` | 20.4548 | 12.1019 | 32.6220 |
| `H2` | 22.0092 | 13.2455 | 34.7316 |
| `H3` | 23.4066 | 13.9832 | 36.5966 |
| `H4` | 24.5300 | 14.6831 | 38.1284 |
| `H5` | 25.5810 | 15.3926 | 39.5005 |
| `H6` | 26.6167 | 15.9251 | 40.8940 |
| `H7` | 27.9178 | 16.7973 | 42.4999 |
| `H8` | 28.9324 | 17.3188 | 43.9034 |
| `H9` | 30.2313 | 18.1884 | 45.5983 |
| `H10` | 31.2636 | 18.8621 | 46.9825 |
| `H11` | 32.5002 | 19.6692 | 48.4932 |
| `H12` | 33.9482 | 20.7855 | 50.4610 |

`gat_gru_h12`

| 步长 | MAE | MAPE(%) | RMSE |
|---|---:|---:|---:|
| `H1` | 18.3109 | 10.8129 | 28.9942 |
| `H2` | 19.8085 | 11.6499 | 31.1601 |
| `H3` | 21.1354 | 12.4415 | 32.9873 |
| `H4` | 22.2759 | 12.9812 | 34.6140 |
| `H5` | 23.3204 | 13.6305 | 36.0352 |
| `H6` | 24.3321 | 14.3155 | 37.4096 |
| `H7` | 25.4458 | 14.8727 | 38.9533 |
| `H8` | 26.4719 | 15.6150 | 40.2755 |
| `H9` | 27.5941 | 16.3447 | 41.7126 |
| `H10` | 28.6031 | 16.8970 | 43.0803 |
| `H11` | 29.7773 | 17.7628 | 44.5412 |
| `H12` | 31.1217 | 18.5327 | 46.3970 |

## 4. 结果分析

### 4.1 精度结论

- `gat_gru_h3` 的总体 `RMSE` 最低，为 `31.2674`，在本轮对照中精度最好。
- `chebnet_gru_h3` 与 `gat_gru_h3` 非常接近，总体 `RMSE` 仅高 `0.0770`，差距很小。
- `gcn_gru_h3` 在三个步长和总体指标上都明显落后于前两者，更适合作为轻量基线模型。
- 在 `h6` 设定下，`gat_gru_h6` 的总体 `RMSE` 最低，为 `33.3330`，仍然保持精度领先。
- `chebnet_gru_h6` 与 `gat_gru_h6` 的总体 `RMSE` 差距为 `0.2294`，依旧比较接近。
- `gcn_gru_h6` 在 `h6` 任务下同样明显落后于前两者。
- 在 `h12` 设定下，`chebnet_gru_h12` 的总体 `RMSE` 为 `38.2440`，略优于 `gat_gru_h12` 的 `38.3708`。
- `chebnet_gru_h12` 与 `gat_gru_h12` 的总体 `MAE` 也几乎相同，说明在更长预测步长下，`ChebNet` 的稳定性开始体现出来。
- `gcn_gru_h12` 在长步长任务下依旧明显落后。

### 4.2 速度结论

- `chebnet_gru_h3` 训练速度和测试速度都明显优于 `gat_gru_h3`。
- `gat_gru_h3` 的训练总耗时约为 `533.71s`，是 `chebnet_gru_h3` 的约 `5.25` 倍。
- `gcn_gru_h3` 虽然比 `gat_gru_h3` 快，但在当前实现下仍慢于 `chebnet_gru_h3`，而且精度也更低。
- 在 `h6` 设定下，`gat_gru_h6` 的训练总耗时达到 `883.50s`，仍然远高于 `ChebNet` 与 `GCN`。
- `chebnet_gru_h6` 的训练时间为 `289.97s`，明显低于 `gat_gru_h6`，但高于 `h3` 设定。
- `gcn_gru_h6` 的测试最快，但综合指标明显弱于 `ChebNet` 和 `GAT`。
- 在 `h12` 设定下，`gat_gru_h12` 训练总耗时 `797.30s`，仍然显著高于 `ChebNet` 与 `GCN`。
- `chebnet_gru_h12` 训练时间为 `224.40s`，比 `gat_gru_h12` 快很多，同时总体精度还略优。
- `gcn_gru_h12` 的训练时间与 `ChebNet` 接近，但精度明显更弱。

### 4.3 工程结论

从“精度、速度、可维护性、本地实验成本”四个维度综合看，当前最适合作为项目主方案的是：

- `ChebNet + GRU`

原因如下：

- 精度接近本轮最佳的 `GAT + GRU`。
- 训练和测试速度显著更快。
- 模块结构相对清晰，后续继续扩展卷积阶数、正则化和输出头时更容易维护。
- 在 `h3`、`h6` 和 `h12` 三组真实实验中都表现出稳定的综合优势。
- 在更长的 `h12` 预测任务中，`ChebNet + GRU` 甚至取得了略优于 `GAT + GRU` 的总体 `RMSE`。

当前更适合作为辅助对照的模型是：

- `GAT + GRU`
  用于说明高表达能力注意力模型的上界性能。
- `GCN + GRU`
  用于说明图卷积基线模型的基本效果与速度表现。

## 5. 论文与答辩建议表述

如果需要把这一轮结果写进论文或答辩材料，可以采用下面的表述逻辑：

1. 在相同数据集和统一训练口径下，对 `GCN`、`ChebNet` 和 `GAT` 三种空间编码器分别开展了 `h3`、`h6` 与 `h12` 多步预测对照实验。
2. 实验结果表明，`GAT` 在预测精度上略优，但其训练和推理开销明显更高。
3. `ChebNet` 在保持较高预测精度的同时，具有更好的训练效率和工程适配性，并在更长预测步长下表现出更稳定的综合效果。
4. 因此，系统最终优先采用 `ChebNet + GRU` 作为主推模型结构，并保留 `GCN` 与 `GAT` 作为对照模型。

## 6. 相关结果文件

本轮关键结果文件如下：

- `results/figures/chebnet_gru_h3_prediction_overview.png`
- `results/figures/gcn_gru_h3_prediction_overview.png`
- `results/figures/gat_gru_h3_prediction_overview.png`
- `results/figures/chebnet_gru_h6_prediction_overview.png`
- `results/figures/gcn_gru_h6_prediction_overview.png`
- `results/figures/gat_gru_h6_prediction_overview.png`
- `results/figures/chebnet_gru_h12_prediction_overview.png`
- `results/figures/gcn_gru_h12_prediction_overview.png`
- `results/figures/gat_gru_h12_prediction_overview.png`
- `results/horizon_metrics/chebnet_gru_h3.json`
- `results/horizon_metrics/gcn_gru_h3.json`
- `results/horizon_metrics/gat_gru_h3.json`
- `results/horizon_metrics/chebnet_gru_h6.json`
- `results/horizon_metrics/gcn_gru_h6.json`
- `results/horizon_metrics/gat_gru_h6.json`
- `results/horizon_metrics/chebnet_gru_h12.json`
- `results/horizon_metrics/gcn_gru_h12.json`
- `results/horizon_metrics/gat_gru_h12.json`
- `results/metrics_summary.csv`

## 7. 下一步实验建议

建议按下面顺序继续推进：

1. 将结果页中的多步趋势图导出并整理成论文插图。
2. 若继续追求精度，可专项优化 `GAT` 的前向效率。
3. 若继续追求综合性价比，可围绕 `ChebNet + GRU` 做主模型调参和结构微调。
4. 可继续补充不同随机种子重复实验，形成均值与方差统计表。
