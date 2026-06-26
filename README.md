# TEGe -- Text Editing or Generating emojis

TEGe, short for **Text Editing or Generating emojis**, is a Track A application-based machine learning project for CS182 in ShanghaiTech. It builds a complete emoji editing/generation-style pipeline where a user provides either a source emoji plus an edit instruction, or only a text description, and the system generates an emoji image while trying to preserve clean emoji style and centered icon composition.

The final prototype is a Gradio demo backed by an InstructPix2Pix diffusion model with LoRA fine-tuning on a curated emoji editing dataset.

## Course Project Alignment

This repository targets Track A from the CS182 Spring 2026 project specification: an application-based ML pipeline with data curation, model selection, experimentation, analysis, and a working prototype.

| Requirement area | How this project addresses it |
| --- | --- |
| Specific application | Natural-language emoji editing and generation-style prompting: `source emoji + text instruction -> edited emoji`, plus a text-only UI mode that uses a neutral emoji seed internally. |
| Data collection and preprocessing | Public Kaggle emoji datasets are downloaded, cleaned, filtered to face emoji, normalized into vendor/canonical image folders, and converted into paired editing examples. |
| Appropriate ML model | The main model is `timbrooks/instruct-pix2pix` with LoRA fine-tuning, which is suitable for instruction-guided image editing. |
| Systematic experimentation | The project compares copy, zero-shot InstructPix2Pix, and LoRA-finetuned variants; it also searches checkpoints and inference guidance settings. |
| Results analysis | Quantitative CLIP metrics and saved visual grids are stored under `artifacts/evaluation/` and `artifacts/quality_search/`. |
| Working prototype | `app.py` launches a Gradio interface for selecting/uploading emoji, entering prompts, and viewing before/after results. |
| Reproducibility | Setup, data, training, inference, evaluation, and artifact paths are documented below. |

The final course package should also include the required final report PDF separately. This README covers the source-code setup and run instructions required by the project specification.

## Problem Definition

Given:

- a source emoji image or the built-in neutral emoji seed,
- an optional source name and vendor style,
- a natural-language edit instruction,

the system generates a target emoji image that satisfies the instruction while preserving as much of the original emoji identity as possible.

The project supports two constructed task types:

- **Semantic edit**: change expression or attributes, such as adding tears, sunglasses, hearts, masks, anger, sadness, or surprise.
- **Style transfer**: render the same emoji identity in another vendor/platform style, such as Apple, Google, Twitter, Samsung, or Facebook.

In the interactive demo, TEGe also provides a **Text-only generation** mode. This mode does not train or invoke a separate pure text-to-image model. Instead, it automatically uses the Apple `neutral face` emoji as a hidden seed source and applies the user's prompt through the same InstructPix2Pix+LoRA editor. This makes it possible to type a description such as "Generate a cheerful yellow emoji with starry eyes" without manually selecting or uploading a source image.

## Repository Structure

```text
.
|-- app.py                                  # Gradio demo UI
|-- emoji_editing/                          # Core data, prompt, inference, and evaluation modules
|-- scripts/
|   |-- download_kaggle_emoji_data.py       # Download public Kaggle sources
|   |-- preprocess_emoji_editing_data.py    # Build curated emoji edit pairs
|   |-- train_emoji_diffusion_editor.py     # LoRA fine-tuning for InstructPix2Pix
|   |-- infer_emoji_editor.py               # Single-image CLI inference
|   |-- evaluate_editor.py                  # Baseline and LoRA quantitative evaluation
|   |-- optimize_editor_quality.py          # Checkpoint/guidance search
|   `-- sweep_guidance.py                   # Guidance-scale sweep helper
|-- data/
|   |-- README.md
|   |-- interim/emoji_editing/metadata/     # Generated CSV metadata and edit pairs
|   `-- processed/emoji_editing/images/     # Processed emoji images
|-- artifacts/
|   |-- emoji_diffusion_editor_60k/          # Main 60k-step LoRA run
|   |-- evaluation/                         # Quantitative evaluation outputs
|   `-- quality_search/                     # Checkpoint/guidance search outputs
`-- requirements-train.txt
```

## Data Pipeline

The project uses public Kaggle emoji resources:

- `subinium/emojiimage-dataset`: vendor-specific emoji images.
- `ajabkhan21/complete-unicode-emoji-dataset-emojis-with-meaning`: Unicode emoji meanings.
- `rtatman/emojinet`: lexical and contextual emoji metadata.
- `shuvokumarbasak4004/emojis-list-unicode-image-dataset`: canonical Unicode emoji images.

The preprocessing pipeline:

1. Downloads and extracts raw Kaggle archives into `data/raw/kaggle/`.
2. Filters to human face emoji and removes animals, objects, and non-face categories.
3. Normalizes vendor emoji images and canonical Unicode targets.
4. Infers emotion and visual attributes from emoji names and metadata.
5. Builds paired examples for semantic editing and style transfer.
6. Writes train/validation/test splits and metadata CSVs.

Current generated dataset summary from `data/interim/emoji_editing/metadata/stats.json`:

| Item | Count |
| --- | ---: |
| Vendor face emoji images | 818 |
| Distinct face emoji | 97 |
| Canonical Unicode images | 94 |
| Style-transfer pairs | 6,386 |
| Semantic-edit pairs | 69,414 |
| Total edit pairs | 75,800 |
| Train / val / test pairs | 68,219 / 3,833 / 3,748 |

## Model and Method

The main model is a LoRA-finetuned version of `timbrooks/instruct-pix2pix`.

Key choices:

- **Base model**: InstructPix2Pix, because the task is naturally instruction-guided image editing.
- **Fine-tuning strategy**: LoRA on UNet attention modules and the text encoder, reducing training cost while adapting the model to emoji-specific style.
- **Training resolution**: 256 x 256.
- **Main run**: 60,000 optimization steps.
- **LoRA rank**: 16.
- **Precision**: `bf16` during training and `fp16` during inference.
- **Default inference settings after quality search**:
  - scheduler: `dpm`
  - steps: `40`
  - text guidance: `3.5`
  - image guidance: `2.8`

The default inference code uses:

```text
artifacts/emoji_diffusion_editor_60k/lora_final
```

This LoRA directory is sufficient for the demo and command-line inference.

## Environment Setup

Use a CUDA-enabled Linux environment for training and practical inference. CPU inference is possible but slow.

```bash
conda create -n emoji-editor python=3.11
conda activate emoji-editor
pip install -r requirements-train.txt
```

If your machine needs a specific CUDA build of PyTorch, install the matching PyTorch wheel first, then install the remaining requirements.

The project has been run locally with an NVIDIA GPU and a CUDA-enabled PyTorch environment. Most scripts keep their configuration in an editable dataclass block near the top of the file.

## Quick Start: Run the Demo

If the processed data and LoRA artifacts are already present, start the Gradio demo directly:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:7860
```

The UI supports:

- text-only generation with a built-in neutral emoji seed,
- choosing a built-in emoji by vendor and emoji name,
- uploading a custom emoji image,
- entering an English edit prompt,
- optional advanced controls for scheduler, seed, resolution, and guidance values,
- side-by-side source and edited output preview.

By default, the app uses GPU device `cuda:0`. To change this, edit `UI_CONFIG.device` in `app.py`.

## Rebuild the Dataset

Run the download script:

```bash
python scripts/download_kaggle_emoji_data.py
```

Run preprocessing:

```bash
python scripts/preprocess_emoji_editing_data.py
```

Main generated files:

```text
data/interim/emoji_editing/metadata/emoji_catalog.csv
data/interim/emoji_editing/metadata/face_emoji_catalog.csv
data/interim/emoji_editing/metadata/vendor_image_index.csv
data/interim/emoji_editing/metadata/style_transfer_pairs.csv
data/interim/emoji_editing/metadata/semantic_edit_pairs.csv
data/interim/emoji_editing/metadata/all_edit_pairs.csv
data/interim/emoji_editing/metadata/stats.json
```

`PREPROCESS_CONFIG.force_rebuild` in `scripts/preprocess_emoji_editing_data.py` controls whether existing generated files are deleted and rebuilt.

## Training

The main diffusion editor is trained with:

```bash
python scripts/train_emoji_diffusion_editor.py
```

Important configuration values are defined in `TRAIN_CONFIG` near the top of the script:

- `pretrained_model_name_or_path`
- `pair_csv`
- `output_dir`
- `resolution`
- `train_batch_size`
- `max_train_steps`
- `learning_rate`
- `rank`
- `mixed_precision`
- `checkpointing_steps`
- `validation_steps`

For multi-GPU training, configure Accelerate and launch:

```bash
accelerate config
accelerate launch scripts/train_emoji_diffusion_editor.py
```

The repository also contains an experimental multimodal conditioner path:

```bash
python scripts/train_multimodal_conditioner.py
```

The final demo path uses the diffusion LoRA editor.

## Command-Line Inference

Edit `INFER_CONFIG` in `scripts/infer_emoji_editor.py`, then run:

```bash
python scripts/infer_emoji_editor.py
```

Default outputs:

```text
artifacts/emoji_editor_output.png
artifacts/emoji_editor_output.json
```

To use a custom image, set `input_image` in `INFER_CONFIG`. If `input_image=None`, the script selects an emoji from the built-in vendor catalog.

The command-line script remains an image-editing entry point. The text-only neutral-seed workflow is implemented in the Gradio frontend in `app.py`.

## Evaluation and Experiments

### Baseline Evaluation

Run:

```bash
python scripts/evaluate_editor.py
```

This compares:

- `copy`: returns the source image unchanged,
- `zeroshot`: base InstructPix2Pix without LoRA,
- `lora`: the fine-tuned emoji editor.

Metrics:

- `clip_text_alignment`: similarity between generated image and edit instruction.
- `clip_image_to_target`: similarity between generated image and constructed target.
- `clip_image_to_source`: preservation of source identity.
- `lpips_to_target`: optional perceptual distance if `lpips` is installed.

Saved outputs:

```text
artifacts/evaluation/editor_metrics.csv
artifacts/evaluation/editor_metrics.json
```

The tracked evaluation artifact reports, on 160 held-out samples, that the LoRA editor improves over zero-shot InstructPix2Pix on target-image similarity overall (`0.9275` vs. `0.8877`) and on semantic-edit target similarity (`0.9567` vs. `0.9070`). The copy baseline has high source preservation but cannot perform the requested edit, so it is mainly a no-op reference.

### Checkpoint and Guidance Search

Run:

```bash
python scripts/optimize_editor_quality.py
```

This script:

1. compares saved 60k LoRA checkpoints,
2. selects the best checkpoint with a CLIP-based composite score,
3. searches scheduler/text-guidance/image-guidance settings,
4. writes result CSVs, JSON summaries, generated samples, and visual grids.

Saved outputs:

```text
artifacts/quality_search/checkpoint_metrics.csv
artifacts/quality_search/checkpoint_comparison.png
artifacts/quality_search/best_checkpoint.json
artifacts/quality_search/guidance_metrics.csv
artifacts/quality_search/guidance_grid.png
artifacts/quality_search/best_guidance.json
```

Current best searched inference setting:

| Setting | Value |
| --- | --- |
| LoRA candidate | `checkpoint-60000/lora` |
| Scheduler | `dpm` |
| Steps | `40` |
| Text guidance | `3.5` |
| Image guidance | `2.8` |
| CLIP text alignment | `0.3042` |
| CLIP target similarity | `0.9589` |
| CLIP source similarity | `0.8970` |
| Composite score | `1.2448` |

`checkpoint-60000/lora` and `lora_final` are equivalent in the tracked checkpoint comparison, so the runnable app keeps `lora_final` as the stable default path.

## Important Artifacts

```text
artifacts/emoji_diffusion_editor_60k/lora_final/
artifacts/emoji_diffusion_editor_60k/train_args.json
artifacts/emoji_diffusion_editor_60k/validation/
artifacts/evaluation/
artifacts/quality_search/
```

The full checkpoint model tensors under:

```text
artifacts/emoji_diffusion_editor_60k/checkpoints/*/model.safetensors
artifacts/emoji_diffusion_editor_60k/checkpoints/*/model_1.safetensors
```

are very large and are ignored by GitHub-oriented commits. They are not required for normal demo inference because the tracked LoRA weights are enough. Keep them locally if full training-resume capability is needed.

## Reproducibility Notes

- The scripts are config-file style: edit the dataclass config block at the top of each script rather than passing many command-line flags.
- Inference defaults currently assume `cuda:0`; change the relevant config field to another device if needed.
- The first run may download the base model and CLIP evaluation model from Hugging Face unless they are already cached.
- Quantitative metrics are useful for comparing settings, but emoji visual quality should also be checked manually with the saved grids.
- The project uses public datasets and open-source pretrained models; final reports or presentations should cite those sources and the base model.

## Minimal Command Checklist

```bash
# 1. Install dependencies
pip install -r requirements-train.txt

# 2. Optional: rebuild data
python scripts/download_kaggle_emoji_data.py
python scripts/preprocess_emoji_editing_data.py

# 3. Run the demo
python app.py

# 4. Run CLI inference
python scripts/infer_emoji_editor.py

# 5. Train LoRA
python scripts/train_emoji_diffusion_editor.py

# 6. Evaluate and tune
python scripts/evaluate_editor.py
python scripts/optimize_editor_quality.py
```
