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

### 2. 修改脚本顶部配置区

现在所有入口脚本都改成了“直接在代码里改配置”的方式，不再依赖项目参数命令行传参。

你只需要打开对应脚本，在文件顶部的配置区修改参数即可。每个字段旁边都写了中文注释，常改的位置主要是：

- [scripts/download_kaggle_emoji_data.py](/Users/ningzd/Desktop/Pic_Gen/scripts/download_kaggle_emoji_data.py)：下载目录、是否强制重下
- [scripts/preprocess_emoji_editing_data.py](/Users/ningzd/Desktop/Pic_Gen/scripts/preprocess_emoji_editing_data.py)：预处理目录、是否强制重建
- [scripts/train_emoji_diffusion_editor.py](/Users/ningzd/Desktop/Pic_Gen/scripts/train_emoji_diffusion_editor.py)：diffusion 训练超参数
- [scripts/train_multimodal_conditioner.py](/Users/ningzd/Desktop/Pic_Gen/scripts/train_multimodal_conditioner.py)：多模态编码器训练超参数
- [scripts/infer_emoji_editor.py](/Users/ningzd/Desktop/Pic_Gen/scripts/infer_emoji_editor.py)：推理脚本配置
- [app.py](/Users/ningzd/Desktop/Pic_Gen/app.py)：Gradio 界面配置

### 3. 下载并预处理数据

```bash
python scripts/download_kaggle_emoji_data.py
python scripts/preprocess_emoji_editing_data.py
```

如果你想重新生成全部中间文件，把 [scripts/preprocess_emoji_editing_data.py](/Users/ningzd/Desktop/Pic_Gen/scripts/preprocess_emoji_editing_data.py) 里的 `force_rebuild` 改成 `True` 再运行一次。

### 4. 单机单卡 RTX 训练

对大多数 RTX 显卡，推理通常优先用 `fp16`，训练可以先试 `bf16`；如果你本机不稳定，再改成 `fp16`。

先在 [scripts/train_emoji_diffusion_editor.py](/Users/ningzd/Desktop/Pic_Gen/scripts/train_emoji_diffusion_editor.py) 顶部改好 `TRAIN_CONFIG`，再直接运行：

```bash
python scripts/train_emoji_diffusion_editor.py
```

单卡常改参数：

- `resolution`
- `train_batch_size`
- `gradient_accumulation_steps`
- `rank`
- `mixed_precision`
- `enable_xformers_memory_efficient_attention`

如果你也要训练多模态条件编码器，先在 [scripts/train_multimodal_conditioner.py](/Users/ningzd/Desktop/Pic_Gen/scripts/train_multimodal_conditioner.py) 里修改 `TRAIN_CONFIG`，然后运行：

```bash
python scripts/train_multimodal_conditioner.py
```

### 5. 推理与交互界面

启动可视化交互界面：

```bash
python app.py
```

命令行推理：

```bash
python scripts/infer_emoji_editor.py
```

如果要直接对本地图片推理，就把 [scripts/infer_emoji_editor.py](/Users/ningzd/Desktop/Pic_Gen/scripts/infer_emoji_editor.py) 顶部的 `input_image` 改成你的图片路径，并把 `instruction` 改成目标编辑要求。

## 多卡集群训练

### 单机多卡服务器

如果你要在单机多卡服务器上训练，脚本内部超参数仍然在代码顶部修改；命令行只负责拉起多进程。

先运行一次：

```bash
accelerate config
```

然后直接启动：

```bash
accelerate launch scripts/train_emoji_diffusion_editor.py
```

如果是多模态编码器的 DDP 训练：

```bash
torchrun --standalone --nproc_per_node=8 scripts/train_multimodal_conditioner.py
```

### 多机多卡集群

如果是多机训练，同样先在脚本顶部配好超参数，再由集群调度器或 `accelerate` 负责分布式拓扑。推荐先在每台机器上完成相同的数据准备和环境安装，然后执行：

```bash
accelerate config
accelerate launch scripts/train_emoji_diffusion_editor.py
```

如果你的集群是通过 Slurm、MPI 或平台侧封装来启动多机任务，也只需要保持脚本本身不带项目参数，交给集群环境注入并行配置即可。
