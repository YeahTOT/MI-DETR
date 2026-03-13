# MI-DETR

**MI-DETR: A Strong Baseline for Moving Infrared Small Target Detection with Bio-Inspired Motion Integration** 的官方代码发布仓库。

本仓库面向论文公开发布进行了整理，保留了论文中使用的核心训练与验证流程，同时移除了本地机器路径、私有运行产物、缓存文件以及不适合上传到 GitHub 的大体积二进制文件。仓库中不包含原始数据集、视网膜预处理后的数据集或大体积预训练权重，仅提供代码、配置文件和下载说明。

## 论文

- 标题：*MI-DETR: A Strong Baseline for Moving Infrared Small Target Detection with Bio-Inspired Motion Integration*
- arXiv: <https://arxiv.org/abs/2603.05071>

```bibtex
@misc{liu2026midetrstrongbaselinemoving,
      title={MI-DETR: A Strong Baseline for Moving Infrared Small Target Detection with Bio-Inspired Motion Integration},
      author={Nian Liu and Jin Gao and Shubo Lin and Yutong Kou and Sikui Zhang and Fudong Ge and Zhiqiang Pu and Liang Li and Gang Wang and Yizheng Wang and Weiming Hu},
      year={2026},
      eprint={2603.05071},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.05071},
}
```

## 概述

MI-DETR 是一个面向运动红外小目标检测的双分支多模态检测器，构建于 RT-DETR 之上。当前公开版本主要聚焦于以下几点：

- 保持与论文一致的模型结构、训练设置和验证流程
- 支持单卡与多卡训练
- 提供便于复现的整洁仓库结构
- 说明环境配置、数据集准备、训练和验证方式

## 模型简介

默认模型配置文件位于 `improve_multimodal/our_resnet18_brain/brain_fuse.yaml`。

关键说明如下：

- 模型使用 6 通道输入。
- `images/` 表示**外观模态**。
- `image/` 表示**运动模态**。
- 在数据加载时，`images/...` 下的每个文件都会与 `image/...` 下同名文件进行配对。
- 数据加载器会将配对后的运动与外观输入拼接成模型使用的 6 通道输入。
- 骨干网络采用外观分支与运动分支组成的双分支结构。
- 在 P3 阶段插入双向跨模态交互模块 `TransformerFusionBlock`。
- 最终融合后的多尺度特征会送入 RT-DETR 解码器。

## 仓库结构

```text
MI-DETR/
├── checkpoints/                  # 权重占位目录及使用说明
├── datasets/                     # 数据集占位目录及结构说明
├── docs/
│   └── DOWNLOADS.md              # 数据集与权重下载说明
├── improve_multimodal/
│   └── our_resnet18_brain/
│       ├── brain_fuse.yaml       # 默认模型配置
│       └── brain_fuse_*.yaml     # 其他配置
├── ultralytics/                  # 基于 Ultralytics 的本地实现
├── data.yaml                     # 默认数据集模板
├── train.py                      # 统一训练入口
├── val.py                        # 统一验证入口
├── requirements.txt
├── .gitignore
└── LICENSE
```

## 环境配置

建议使用 Python 3.10 或 3.11。如果使用 CUDA，请先安装与你本地 CUDA 环境匹配的 PyTorch 和 torchvision，再安装其余依赖。

```bash
conda create -n midetr python=3.10 -y
conda activate midetr

pip install --upgrade pip
pip install -r requirements.txt
```

## 数据集准备

### 1. 原始数据集

DAUB-R、ITSDT-15K 和 IRDST-H 的原始数据下载链接请参考 MoPKL 仓库：

- <https://github.com/UESTC-nnLab/MoPKL>

### 2. 视网膜预处理数据集与权重

为了复现实验结果，推荐使用视网膜预处理后的数据集以及公开发布的权重。

- 文件名：`Dataset_retina`
- 百度网盘：<https://pan.baidu.com/s/1p5409A7rldXrFzzcwC_ALQ?pwd=5paw>
- 提取码：`5paw`

更多说明见 [docs/DOWNLOADS.md](./docs/DOWNLOADS.md)。

### 3. 期望的数据集目录结构

默认目录结构如下：

```text
datasets/
└── DAUB-R_retina/
    ├── images/
    │   ├── train/
    │   └── test/
    ├── image/
    │   ├── train/
    │   └── test/
    └── labels/
        ├── train/
        └── test/
```

重要说明：

- `images/` 是**外观模态**目录。
- `image/` 是**运动模态**目录。
- `images/` 和 `image/` 下的文件名必须严格一一对应。
- 加载器会自动将 `images/.../xxx.png` 映射到 `image/.../xxx.png`。
- 标注采用标准 YOLO 检测格式。
- 默认 `data.yaml` 使用 `images/test` 作为验证集划分。

### 4. 如何使用 `data.yaml`

默认数据集模板为 `data.yaml`，常见使用方式有两种：

1. 将数据集放在仓库内的 `datasets/` 目录下，直接使用 `data.yaml`。
2. 将数据集放在机器上的任意位置，并通过 `--dataset-root` 覆盖数据集根目录。

示例：

```bash
python train.py --data data.yaml --dataset-root /path/to/DAUB-R_retina --device 0
```

如果要复现 ITSDT-15K 或 IRDST-H，只需将 `--dataset-root` 切换到对应的视网膜预处理数据集目录。如果你的本地划分名称与默认模板不同，可复制 `data.yaml` 后自行修改。

### 5. 使用 `motion_map.py` 生成运动图

如果你当前只有 `images/` 下的外观帧，可以使用 `motion_map.py` 离线生成对应的 `image/` 运动模态目录。

常用示例：

```bash
python motion_map.py \
  --source-root /path/to/DAUB-R_retina/images \
  --output-root /path/to/DAUB-R_retina/image \
  --recursive
```

说明：

- `--source-root` 通常指向外观图像目录，即 `images/`。
- `--output-root` 通常指向运动图输出目录，即 `image/`。
- `--recursive` 会递归处理子目录，适合同时生成 `train/` 和 `test/` 下的运动图。
- `--mode reference` 使用论文参考实现。
- `--mode paper_onnx` 使用 ONNX-friendly 的递推 RCA core，并在宿主侧执行论文中的 bilateral enhance；保存出的运动图与 `reference` 对齐。
- `--mode onnx_approx` 使用纯图内近似版，便于整图导出和集成部署，但不保证与论文运动图像素级一致，视觉上通常会更亮一些。
- 默认会以 3 通道 PNG 保存运动图；如果需要单通道输出，可加 `--no-save-rgb`。

如果你需要在“贴近论文”和“纯 ONNX 近似”之间显式选择，推荐分别使用下面两种命令：

```bash
python motion_map.py \
  --source-root /path/to/DAUB-R_retina/images \
  --output-root /path/to/DAUB-R_retina/image \
  --recursive \
  --mode paper_onnx
```

```bash
python motion_map.py \
  --source-root /path/to/DAUB-R_retina/images \
  --output-root /path/to/DAUB-R_retina/image_approx \
  --recursive \
  --mode onnx_approx
```

## 训练

`train.py` 是统一的公开训练入口，不包含任何本地绝对路径或固定 GPU ID。

### 单卡训练

```bash
python train.py \
  --data data.yaml \
  --dataset-root /path/to/DAUB-R_retina \
  --device 0 \
  --batch 32 \
  --epochs 600 \
  --imgsz 512 \
  --name daub-r
```

### 多卡训练

```bash
python train.py \
  --data data.yaml \
  --dataset-root /path/to/DAUB-R_retina \
  --device 0,1 \
  --batch 32 \
  --epochs 600 \
  --imgsz 512 \
  --name daub-r_ddp
```

说明：

- `--device 0` 表示单卡训练。
- `--device 0,1` 表示通过 Ultralytics DDP 进行多卡训练。
- `--batch` 是全局 batch size，在 DDP 模式下会自动分配到各张 GPU。
- 默认脚本保留了公开版本中使用的主要训练设置，包括 `AdamW`、`imgsz=512`、`epochs=600` 和 `close_mosaic=80`。

## 验证

将公开权重放入 `checkpoints/`，或者通过 `--weights` 传入外部路径。

```bash
python val.py \
  --weights checkpoints/ITSDT-15K.pt \
  --data data.yaml \
  --dataset-root /path/to/ITSDT-15K_retina \
  --device 0 \
  --imgsz 512 \
  --batch 1 \
  --name ITSDT-15K_val
```

对于其他数据集，只需切换以下参数：

- `--weights`
- `--dataset-root`
- 如有需要还可切换 `--data`

## 单张图像预测

使用 `pred.py` 对单张图像进行推理。输入路径必须指向 `images/` 目录下的外观图像，脚本会自动在对应的 `image/` 路径中查找同名运动图像。

```bash
python pred.py \
  --weights checkpoints/DAUB-R.pt \
  --source /path/to/DAUB-R_retina/images/test/000001.png \
  --device 0 \
  --imgsz 512 \
  --conf 0.25 \
  --name daub-r_pred
```

输出保存在 `runs/pred/<name>`，内容包括：

- 来自外观分支和运动分支的可视化预测图像
- 包含类别 ID、类别名、置信度以及像素坐标系 `xyxy` 框的 JSON 文件
- 终端中的检测结果摘要

## ONNX 导出

使用 `export.py` 将公开的 PyTorch 权重导出为 ONNX。

```bash
python export.py \
  --weights checkpoints/DAUB-R.pt \
  --output checkpoints/DAUB-R.onnx \
  --imgsz 512
```

该功能依赖 `onnx`。如果启用了图优化简化，还需要安装 `onnxruntime`。

如果要导出集成流式运动分支的 ONNX，请启用 `--with-motion-stream`。这种模型会将检测输出保留在同一个文件内，同时把运动生成并入图中，并暴露用于部署的循环状态输入输出：

```bash
python export.py \
  --weights checkpoints/DAUB-R.pt \
  --with-motion-stream \
  --output checkpoints/DAUB-R-stream.onnx \
  --imgsz 512
```

集成流式运动分支的 ONNX 接口如下：

- 输入：`frame`、`adapt_state`、`memory_state`、`state_valid`
- 输出：`output0`、`next_adapt_state`、`next_memory_state`
- `frame` 是预处理后的 `1x3xHxW` 浮点张量，不是任意原始分辨率图像
- 该模型仍然是有状态的，因此部署代码需要持续回传 `next_*` 状态
- 集成流式模型内部使用的是 `onnx_approx` 运动分支，适合单图部署，但其运动图外观不追求与论文 reference 像素级一致

如果你想单独导出运动分支的递推 core，请使用 `export_mition_onnx.py`：

```bash
python export_mition_onnx.py \
  --output checkpoints/motion_map.onnx \
  --imgsz 512
```

该导出产物暴露的是 RCA core，而不是最终增强后的运动图：

- 输入：`frame`、`adapt_state`、`memory_state`、`state_valid`
- 输出：`motion_core`、`next_adapt_state`、`next_memory_state`
- 如果希望得到与论文一致的最终运动图，需要在 ONNX 输出的 `motion_core` 之后，在宿主侧再执行一次论文中的 `paper_postprocess`（幂次增强 + bilateral + 按帧归一化）

## ONNX 帧序列预测

使用 `video_onnx.py` 对一个帧目录执行 ONNX 推理。脚本会将外观帧复制到运行目录中，在 `input/image/` 下生成配对的运动图，并保存带标注的检测结果图像。

```bash
python video_onnx.py \
  --weights checkpoints/DAUB-R.onnx \
  --source /home/tot/project/VT5025-2512/obj_det/MI-DETR/datasets/infers/1 \
  --motion-mode paper_onnx \
  --imgsz 512 \
  --conf 0.25 \
  --name daub-r_onnx
```

说明：

- `video_onnx.py` 仅接受帧目录，不支持 MP4 文件。
- `video.py` 仍然是 PyTorch 权重和 MP4 工作流的入口。
- `--motion-mode` 支持 `reference`、`paper_onnx`、`onnx_approx`，默认是 `reference`。
- 如果你在 ONNX 检测前希望生成最接近论文的运动图，优先使用 `paper_onnx`。
- 如果你只是为了复用纯图内近似分支或与集成流式导出保持一致，可使用 `onnx_approx`。
- ONNX 推理依赖 `onnx` 和 `onnxruntime`。

## ONNX 流式预测

当你需要模拟真实流式路径时，使用 `video_onnx_stream.py`：每一帧都会先更新循环运动状态，再立即对当前帧执行 ONNX 检测。它同时支持：

- `DAUB-R.onnx`：仅检测器 ONNX，由 Python 先生成运动信息
- `DAUB-R-stream.onnx`：集成流式运动分支的 ONNX，由图内部完成运动生成

```bash
python video_onnx_stream.py \
  --weights checkpoints/DAUB-R.onnx \
  --source datasets/infers/1 \
  --conf 0.25 \
  --name daub-r_onnx_stream

```
```bash
python video_onnx_stream.py  --weights checkpoints/DAUB-R-stream.onnx   --source datasets/infers/1   --conf 0.25   --name daub-r_onnx_stream_integrated
```


说明：

- `video_onnx_stream.py` 仅支持帧目录。
- `video_onnx.py` 会先为整个序列生成运动图，再进行批量预测。
- 当输入是 `DAUB-R.onnx` 时，`video_onnx_stream.py` 会逐帧运行 paper-aligned host pipeline，并在进入新的子目录时重置运动状态。
- 当输入是 `DAUB-R-stream.onnx` 时，运动生成发生在图内部，走的是 `onnx_approx` 分支，因此更适合部署一致性，而不是论文外观一致性。
- 脚本结束时会打印端到端 FPS 汇总，覆盖逐帧运动生成、推理和结果保存。
- 输出结果会以带标注图像保存在 `images/` 下，以逐帧 JSON 保存在 `json/` 下。

## 复现说明

为尽可能接近论文设置，建议：

- 使用视网膜预处理后的数据集
- 保持 `images/` 为外观模态，`image/` 为运动模态
- 保证两个模态目录之间严格一一配对
- 使用默认模型配置 `improve_multimodal/our_resnet18_brain/brain_fuse.yaml`
- 保持 `imgsz=512`、`epochs=600` 和 `optimizer=AdamW`
- 使用公开权重或训练得到的 `best.pt` 进行验证

默认情况下，输出保存在：

- 训练：`runs/train/<name>`
- 验证：`runs/val/<name>`

## 开源发布说明

当前公开仓库已做如下清理：

- 移除本地绝对路径
- 统一训练与验证入口
- 移除缓存、压缩包、临时运行结果和大体积权重文件
- 保留论文复现所需的核心代码路径

## 许可证

本仓库包含修改后的、基于 Ultralytics 的实现。为了与上游许可基础保持一致，本仓库以 **AGPL-3.0** 协议发布。详情见 [LICENSE](./LICENSE)。

## 引用

如果你觉得本仓库对你的工作有帮助，请引用：

```bibtex
@misc{liu2026midetrstrongbaselinemoving,
      title={MI-DETR: A Strong Baseline for Moving Infrared Small Target Detection with Bio-Inspired Motion Integration},
      author={Nian Liu and Jin Gao and Shubo Lin and Yutong Kou and Sikui Zhang and Fudong Ge and Zhiqiang Pu and Liang Li and Gang Wang and Yizheng Wang and Weiming Hu},
      year={2026},
      eprint={2603.05071},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.05071},
}
```
