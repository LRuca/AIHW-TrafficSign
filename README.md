# Traffic Sign Experiments

这是一个基于 PyTorch 的交通标志分类实验项目，支持配置驱动的实验管理、单实验训练、批量运行、结果汇总，以及本地 dashboard 的监控、排队、归档和断点恢复。

项目默认面向 GTSRB 风格的数据目录组织方式。

## 1. 项目结构

```text
configs/
  base.yaml                 全局默认配置
  experiment_plan.yaml      dashboard 实验计划
  experiments/              每个具体实验的 yaml

scripts/
  run_experiment.py         跑单个实验
  run_stage.py              跑一个目录下的所有实验
  summarize_results.py      汇总 outputs/runs 下的结果
  launch_dashboard.py       dashboard 后端
  start_dashboard.ps1       一键启动 dashboard
  run_in_pytorch.ps1        使用指定 conda 环境运行 Python 脚本

src/traffic_signs/
  data.py                   数据读取与增强
  models.py                 模型构建
  train.py                  训练、保存、恢复
  analysis.py               曲线图、混淆矩阵、指标导出
  config.py                 配置加载

web/
  index.html                dashboard 前端

data/                       数据集目录
outputs/                    训练输出、日志、图表、队列状态
```

## 2. 环境配置

### 2.1 推荐方式：使用项目内置环境脚本

当前项目默认使用：

```text
C:\Users\lenovo\.conda\envs\pytorch\python.exe
```

[scripts/run_in_pytorch.ps1](/C:/Users/lenovo/Desktop/RJ/FinalHW/scripts/run_in_pytorch.ps1:1) 会自动：

- 使用指定 Python
- 补齐 `PATH`
- 设置 `PYTHONPATH=<项目根目录>\src`

先验证环境是否正常：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_in_pytorch.ps1
```

如果成功，会打印 `python ok`、`numpy`、`torch`、`torchvision` 的版本。

### 2.2 通用方式：自己创建环境

如果你不想绑定当前 conda 环境，也可以自己创建 Python 环境，然后安装依赖：

```powershell
pip install -r requirements.txt
```

再设置：

```powershell
$env:PYTHONPATH="$PWD\src"
```

然后可以直接运行：

```powershell
python scripts\run_experiment.py --config configs\experiments\backbone_mobilenetv2.yaml
```

## 3. 依赖文件

项目根目录已经提供：

- `requirements.txt`
- `requirement.txr`

这两个文件内容相同，只是兼容你当前的命名习惯。

当前项目里实际验证过的一组版本是：

```text
Python 3.6.13
numpy 1.19.2
pandas 1.1.5
matplotlib 3.3.4
Pillow 8.3.1
PyYAML 6.0.1
timm 0.6.12
torch 1.10.2
torchvision 0.11.3
```

`requirements.txt` 目前也按这组版本固定。

## 4. 数据集准备

### 4.1 默认目录结构

默认配置见 [configs/base.yaml](/C:/Users/lenovo/Desktop/RJ/FinalHW/configs/base.yaml:1)，数据目录为：

```text
data/train_images/
data/val_images/
data/test_images/
```

每个 split 下面按类别建文件夹，例如：

```text
data/train_images/00/
data/train_images/01/
...
data/train_images/42/
```

支持的图片格式：

- `.png`
- `.jpg`
- `.jpeg`
- `.bmp`

### 4.2 验证集来源

[configs/base.yaml](/C:/Users/lenovo/Desktop/RJ/FinalHW/configs/base.yaml:14) 默认：

```yaml
use_existing_val_dir: true
```

这表示：

- 如果 `data/val_images` 存在，就直接使用它做验证集
- 如果改成 `false`，代码会从 `train_images` 中按 `train_split` 自动划分训练集和验证集

## 5. 配置实验

### 5.1 基础配置：`configs/base.yaml`

这个文件定义所有实验共享的默认值，例如：

- 数据路径
- 图像尺寸
- epoch 数
- batch size
- 模型默认类型
- optimizer 和 scheduler 默认设置
- 输出目录

常用字段示例：

```yaml
output_root: outputs/runs

data:
  train_dir: data/train_images
  val_dir: data/val_images
  test_dir: data/test_images
  num_classes: 43
  image_size: 64
  num_workers: 4

train:
  seed: 42
  device: cuda
  batch_size: 64
  epochs: 30
  amp: true

model:
  name: mobilenet_v2
  pretrained: true
  dropout: 0.2
  activation: relu

finetune:
  mode: full

augmentation:
  policy: basic

optimizer:
  name: adamw
  lr: 0.001

scheduler:
  name: cosine
```

### 5.2 单个实验配置：`configs/experiments/*.yaml`

每个实验文件只需要覆盖你想改的字段。例如：

```yaml
experiment_name: stage1_backbone_mobilenetv2
metadata:
  stage: backbone
  dimension: model.name
  value: mobilenet_v2

model:
  name: mobilenet_v2
```

你最常改的字段通常有：

- `experiment_name`
- `metadata.stage`
- `metadata.dimension`
- `metadata.value`
- `model.name`
- `data.image_size`
- `train.epochs`
- `finetune.mode`
- `augmentation.policy`
- `optimizer.name`
- `scheduler.name`

### 5.3 新增一个实验配置的完整示例

下面是一个完整示例：新增一个 96 分辨率、40 个 epoch、使用 AdamW 和 cosine scheduler 的 DeiT tiny 实验。

你可以新建文件：

```text
configs/experiments/custom_deit_96_ep40.yaml
```

内容如下：

```yaml
experiment_name: custom_deit_96_ep40
metadata:
  stage: custom
  dimension: model.name
  value: deit_tiny_patch16_224

data:
  image_size: 96

train:
  epochs: 40
  batch_size: 64
  seed: 42
  device: cuda
  amp: true

model:
  name: deit_tiny_patch16_224
  pretrained: true
  activation: gelu
  dropout: 0.2

finetune:
  mode: full

augmentation:
  policy: basic_color

optimizer:
  name: adamw
  lr: 0.0005
  weight_decay: 0.0001
  momentum: 0.9

scheduler:
  name: cosine
  min_lr: 0.000001
  gamma: 0.97
  power: 2.0

runtime:
  save_predictions: true
  save_confusion_matrix: true
  save_curves: true
  save_classwise_metrics: true
```

建好之后，你可以直接运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_in_pytorch.ps1 scripts\run_experiment.py --config configs\experiments\custom_deit_96_ep40.yaml
```

如果你还想让它出现在 dashboard 的“实验计划”和“可选实验”里，需要继续把它加入 [configs/experiment_plan.yaml](/C:/Users/lenovo/Desktop/RJ/FinalHW/configs/experiment_plan.yaml:1)：

```yaml
- stage: custom
  dimension: model.name
  description: 自定义实验
  experiments:
    - experiment_name: custom_deit_96_ep40
      value: deit_tiny_patch16_224
```

### 5.4 dashboard 实验计划：`configs/experiment_plan.yaml`

这个文件决定 dashboard 的“实验计划 / 可选实验 / 横向比较”如何显示。

如果你新增了实验配置，想让它出现在 dashboard 中，需要：

1. 在 `configs/experiments/` 下新增 yaml
2. 在 `configs/experiment_plan.yaml` 里把它加到某个 stage 的 `experiments` 中

## 6. 如何开始运行实验

### 6.1 跑单个实验

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_in_pytorch.ps1 scripts\run_experiment.py --config configs\experiments\backbone_mobilenetv2.yaml
```

### 6.2 批量跑一个目录下的实验

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_in_pytorch.ps1 scripts\run_stage.py --config-dir configs\experiments
```

### 6.3 汇总实验结果

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_in_pytorch.ps1 scripts\summarize_results.py
```

默认输出到：

```text
outputs/summaries/experiment_summary.csv
```

## 7. 输出文件说明

每次实验都会在 `outputs/runs/` 下生成一个 run 目录，常见文件有：

- `status.json`
- `train_log.csv`
- `best.pt`
- `last.pt`
- `resume_checkpoint.pt`
- `metrics.json`
- `curves.png`
- `lr_curve.png`
- `confusion_matrix.png`
- `classwise_metrics.csv`
- `val_predictions.csv`
- `test_predictions.csv`

## 8. 如何启动 dashboard

### 8.1 推荐方式

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_dashboard.ps1
```

然后访问：

```text
http://127.0.0.1:8765/
```

### 8.2 直接运行后端

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_in_pytorch.ps1 scripts\launch_dashboard.py --host 127.0.0.1 --port 8765
```

## 9. Dashboard 怎么用

dashboard 主要有三个页签：

- 单实验详情
- 实验计划与横向比较
- 实验队列

实验队列的使用流程：

1. 在左侧“可选实验”里点击实验卡片，把实验加入队列
2. 在右侧队列中调整顺序或删除条目
3. 点击“启动队列”
4. dashboard 会按顺序依次运行实验

## 10. 中断与恢复

在 dashboard 的“实验队列”页点击“停止当前实验”后，当前实验会尽量优雅停止，并在当前 run 目录中保存：

- `last.pt`
- `resume_checkpoint.pt`
- `status.json` 中的 `status: stopped`

恢复方式：

1. 再次把同一个实验加入队列
2. 点击“启动队列”

dashboard 会自动复用最近一次 `stopped` 的同名实验目录并继续训练。

## 11. 常见问题

### 11.1 dashboard 页面打不开

优先检查：

- `scripts/start_dashboard.ps1` 是否报错
- `outputs/logs/dashboard_stdout.log`
- `outputs/logs/dashboard_stderr.log`
- 8765 端口是否被占用

### 11.2 训练启动后立刻报错

优先检查：

- 数据目录是否存在
- 类别文件夹是否完整
- `num_classes` 是否正确
- 模型名是否受支持
- 当前环境是否已安装 `torch`、`torchvision`、`timm`
