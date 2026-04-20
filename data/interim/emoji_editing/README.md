This folder contains cleaned metadata and generated edit pairs for the emoji editing project.

## Files

- `metadata/emoji_catalog.csv`: merged metadata across Kaggle sources
- `metadata/face_emoji_catalog.csv`: filtered human-face emoji subset with heuristic attributes
- `metadata/vendor_image_index.csv`: one row per curated vendor image
- `metadata/style_transfer_pairs.csv`: same emoji, different vendor style pairs
- `metadata/semantic_edit_pairs.csv`: same vendor, different target emoji pairs
- `metadata/all_edit_pairs.csv`: combined training pair index
- `metadata/stats.json`: summary counts for the curated subset

Pair CSV files include a `split` column with deterministic `train` / `val` / `test` assignments.

Processed image assets are written under `data/processed/emoji_editing/`.
