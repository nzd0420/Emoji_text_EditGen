"""Core package for the emoji editing project."""

from .catalog import (
    EmojiCatalogEntry,
    catalog_lookup,
    entries_for_vendor,
    load_vendor_catalog,
    vendors_from_catalog,
)
from .diffusion_data import EmojiDiffusionCollator, EmojiDiffusionEditDataset
from .diffusion_inference import (
    choices_for_vendor,
    edit_emoji_image,
    load_editor_pipeline,
    load_ui_catalog,
)
from .multimodal_data import (
    EmojiEditCollator,
    EmojiEditMultimodalDataset,
    MultimodalBatch,
    MultimodalLabelVocab,
    build_label_vocab_from_csv,
    save_label_vocab,
)
from .multimodal_model import (
    EmojiEditMultimodalConfig,
    EmojiEditMultimodalEncoder,
)

__all__ = [
    "EmojiEditCollator",
    "EmojiDiffusionCollator",
    "EmojiDiffusionEditDataset",
    "EmojiEditMultimodalConfig",
    "EmojiEditMultimodalDataset",
    "EmojiEditMultimodalEncoder",
    "EmojiCatalogEntry",
    "MultimodalBatch",
    "MultimodalLabelVocab",
    "build_label_vocab_from_csv",
    "catalog_lookup",
    "choices_for_vendor",
    "edit_emoji_image",
    "entries_for_vendor",
    "load_editor_pipeline",
    "load_ui_catalog",
    "load_vendor_catalog",
    "save_label_vocab",
    "vendors_from_catalog",
]
