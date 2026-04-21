# Emoji Editing Generation

本项目目标是实现一个基于参考图片与文本指令的 Emoji 编辑生成系统。

用户上传一个原始 emoji，并输入一段自然语言修改要求，模型输出一个新的 emoji 图像。生成结果需要尽量保留原始 emoji 的基础风格与主体特征，同时根据指令完成目标编辑。

## 预期功能

- 支持 `image + text -> image` 的多模态 emoji 编辑任务
- 输入一张参考 emoji，输出符合文本要求的新 emoji
- 支持表情修改，例如开心、难过、生气、惊讶、哭泣等
- 支持局部属性编辑，例如眼镜、眼泪、爱心、嘴型、眼型等视觉元素变化
- 支持风格迁移，例如在不同平台 emoji 风格之间进行转换
- 在编辑过程中尽量保持原始 emoji 的整体身份、构图与配色一致性
- 支持基于数据集自动构造训练样本，用于完成可控编辑生成任务

## 项目目标

本项目希望构建一个小而完整的生成系统，让用户能够通过“上传 emoji + 输入修改要求”的方式，得到自然、清晰、可控的新 emoji 结果，并验证多模态条件生成在受限视觉域中的实际效果。

## 使用步骤

### 1. 安装依赖

```bash
pip install -r requirements-train.txt
```

### 2. 下载并预处理数据

```bash
python scripts/download_kaggle_emoji_data.py
python scripts/preprocess_emoji_editing_data.py --force
```

### 3. 单机单卡 RTX 训练

对大多数 RTX 显卡，`fp16` 是最稳妥的默认选择。

```bash
accelerate launch --num_processes 1 --mixed_precision fp16 \
  scripts/train_emoji_diffusion_editor.py \
  --output-dir artifacts/emoji_diffusion_editor \
  --resolution 256 \
  --train-batch-size 24 \
  --gradient-accumulation-steps 2 \
  --gradient-checkpointing \
  --allow-tf32 \
  --train-text-encoder-lora
```

如果你的显卡显存更大，可以适当增大：

- `--train-batch-size`
- `--resolution`
- `--rank`

如果已经安装 `xformers`，可以额外加入：

```bash
--enable-xformers-memory-efficient-attention
```

### 4. 推理与交互界面

启动可视化交互界面：

```bash
python app.py \
  --base-model timbrooks/instruct-pix2pix \
  --lora-path artifacts/emoji_diffusion_editor/lora_final
```

命令行推理：

```bash
python scripts/infer_emoji_editor.py \
  --instruction "Add sunglasses and make the face more confident." \
  --vendor Apple
```

如果要直接对本地图片推理：

```bash
python scripts/infer_emoji_editor.py \
  --input-image path/to/emoji.png \
  --instruction "Turn this into a crying emoji with visible tears."
```

## 多卡集群训练

### 单机多卡服务器

例如单机 8 卡：

```bash
accelerate launch --multi_gpu --mixed_precision bf16 \
  --num_processes 8 \
  scripts/train_emoji_diffusion_editor.py \
  --output-dir artifacts/emoji_diffusion_editor \
  --resolution 256 \
  --train-batch-size 24 \
  --gradient-accumulation-steps 1 \
  --gradient-checkpointing \
  --allow-tf32 \
  --train-text-encoder-lora \
  --enable-xformers-memory-efficient-attention
```

### 多机多卡集群

如果是多机训练，先在各节点完成相同环境与数据准备，然后设置分布式环境变量，再运行：

```bash
accelerate launch --multi_gpu --mixed_precision bf16 \
  --num_processes 8 \
  --num_machines 2 \
  --machine_rank 0 \
  --main_process_ip <MASTER_ADDR> \
  --main_process_port <MASTER_PORT> \
  scripts/train_emoji_diffusion_editor.py \
  --output-dir artifacts/emoji_diffusion_editor \
  --resolution 256 \
  --train-batch-size 24 \
  --gradient-accumulation-steps 1 \
  --gradient-checkpointing \
  --allow-tf32 \
  --train-text-encoder-lora \
  --enable-xformers-memory-efficient-attention
```

第二台机器将 `--machine_rank` 改为 `1`，其余参数保持一致。
