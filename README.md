# Traffic Flow Prediction

基于图神经网络的交通流量预测项目，当前同时支持命令行训练、批量实验和 PyQt 图形界面。

## 当前结构

```text
traffic_flow_prediction/
├─ src/                     # 核心源码
│  ├─ datasets/             # 数据读取与图构建
│  ├─ models/               # 时空模型定义
│  ├─ trainers/             # 训练与测试流程
│  ├─ services/             # GUI 业务服务
│  ├─ gui/                  # 界面与页面组件
│  ├─ workers/              # GUI 后台任务
│  ├─ utils/                # 通用工具
│  └─ project_paths.py      # 项目路径解析
├─ configs/                 # 数据 / 训练 / 模型配置
├─ data/                    # 原始数据
├─ results/                 # 训练结果、图像、报告
├─ train.py                 # 训练入口
├─ run_gui.py               # GUI 入口
├─ run_all.py               # 批量实验入口
├─ rungui.bat               # Windows 双击启动 GUI
└─ build_exe.bat            # Windows 打包脚本
```

## 运行方式

安装依赖：

```bash
pip install -r requirements.txt
```

启动 GUI：

```bash
python run_gui.py
```

训练单个模型：

```bash
python train.py --model_cfg configs/model/gcn_gru.yaml
```

执行批量实验：

```bash
python run_all.py
```

## 配置目录

- `configs/data/*.yaml`：数据集路径、节点数、训练/测试天数、时间间隔
- `configs/train/*.yaml`：训练超参数、设备、早停、损失函数等
- `configs/model/*.yaml`：图构建方式、空间模块、时间模块、输出头等

## 打包说明

推荐使用仓库内的脚本，而不是手写 `pyinstaller` 命令：

```bash
build_exe.bat
```

常见形式：

```bash
build_exe.bat onedir
build_exe.bat onefile
build_exe.bat onedir lite
build_exe.bat onefile full
```

## 这次重构后的约定

- 统一使用 `src.*` 包导入，不再依赖修改 `sys.path`
- 运行时路径统一按项目根目录解析，避免从不同目录启动时找不到配置或数据
- 根目录只保留真正对外的入口脚本，核心实现都放在 `src/`
- 历史遗留入口已废弃，不应继续作为主流程使用
