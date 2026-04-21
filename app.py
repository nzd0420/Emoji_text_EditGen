"""Interactive Gradio UI for the emoji diffusion editor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gradio as gr
from PIL import Image

from emoji_editing.catalog import EmojiCatalogEntry
from emoji_editing.diffusion_inference import choices_for_vendor, edit_emoji_image, load_ui_catalog
from emoji_editing.prompting import DEFAULT_NEGATIVE_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="timbrooks/instruct-pix2pix")
    parser.add_argument("--lora-path", default="artifacts/emoji_diffusion_editor/lora_final")
    parser.add_argument("--vendor-index-csv", default="data/interim/emoji_editing/metadata/vendor_image_index.csv")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def build_interface(args: argparse.Namespace) -> gr.Blocks:
    catalog_bundle = load_ui_catalog(args.vendor_index_csv)
    entries: list[EmojiCatalogEntry] = catalog_bundle["entries"]
    lookup: dict[str, EmojiCatalogEntry] = catalog_bundle["lookup"]
    vendors: list[str] = catalog_bundle["vendors"]
    default_vendor = vendors[0]
    default_choices = choices_for_vendor(entries, default_vendor)
    default_key = default_choices[0][1]
    default_entry = lookup[default_key]

    def load_preview(path: str) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGBA").copy()

    css = """
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');
    :root {
      --bg-main: #f5efe2;
      --bg-panel: #fffaf1;
      --ink-main: #1e1d1a;
      --ink-soft: #5d594f;
      --accent-main: #df5b2d;
      --accent-soft: #0f7a7a;
      --border-soft: rgba(30, 29, 26, 0.12);
    }
    .gradio-container {
      font-family: "Space Grotesk", sans-serif !important;
      background:
        radial-gradient(circle at top left, rgba(223, 91, 45, 0.16), transparent 26%),
        radial-gradient(circle at bottom right, rgba(15, 122, 122, 0.14), transparent 24%),
        var(--bg-main);
      color: var(--ink-main);
    }
    .hero {
      padding: 18px 22px;
      border: 1px solid var(--border-soft);
      border-radius: 22px;
      background: linear-gradient(135deg, rgba(255,250,241,0.96), rgba(246,238,220,0.94));
      box-shadow: 0 18px 40px rgba(34, 31, 24, 0.08);
      margin-bottom: 12px;
    }
    .hero h1 {
      margin: 0;
      font-size: 2.2rem;
      line-height: 1.05;
      letter-spacing: -0.04em;
    }
    .hero p {
      margin: 10px 0 0 0;
      color: var(--ink-soft);
      font-size: 1rem;
    }
    .panel {
      border: 1px solid var(--border-soft);
      border-radius: 20px;
      background: rgba(255, 250, 241, 0.94);
      box-shadow: 0 12px 30px rgba(34, 31, 24, 0.06);
    }
    .meta-box {
      padding: 12px 14px;
      border-radius: 16px;
      background: rgba(15, 122, 122, 0.08);
      color: var(--ink-main);
      font-size: 0.95rem;
      min-height: 56px;
    }
    """

    def update_emoji_choices(vendor: str):
        vendor_choices = choices_for_vendor(entries, vendor)
        selected_key = vendor_choices[0][1]
        entry = lookup[selected_key]
        return (
            gr.Dropdown(choices=vendor_choices, value=selected_key),
            load_preview(entry.image_path),
            f"Selected: {entry.display_name} in {entry.vendor} style",
        )

    def update_preview(selected_key: str):
        entry = lookup[selected_key]
        return load_preview(entry.image_path), f"Selected: {entry.display_name} in {entry.vendor} style"

    def run_edit(
        source_mode: str,
        uploaded_image: Image.Image | None,
        selected_key: str,
        instruction: str,
        negative_prompt: str,
        steps: int,
        guidance_scale: float,
        image_guidance_scale: float,
        seed: int,
        resolution: int,
        extra_style_hint: str,
        scheduler_name: str,
    ):
        if not instruction.strip():
            raise gr.Error("请输入编辑要求。")

        source_name = None
        source_vendor = None
        if source_mode == "Upload":
            if uploaded_image is None:
                raise gr.Error("请先上传一个 emoji 图像。")
            source_image = uploaded_image.convert("RGBA")
        else:
            entry = lookup[selected_key]
            source_image = load_preview(entry.image_path)
            source_name = entry.name
            source_vendor = entry.vendor

        result, metadata = edit_emoji_image(
            source_image=source_image,
            instruction=instruction,
            base_model=args.base_model,
            lora_path=args.lora_path if Path(args.lora_path).exists() else None,
            precision=args.precision,
            device=args.device,
            source_name=source_name,
            source_vendor=source_vendor,
            steps=steps,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
            negative_prompt=negative_prompt,
            seed=int(seed),
            resolution=int(resolution),
            scheduler_name=scheduler_name,
            extra_style_hint=extra_style_hint.strip() or None,
        )
        return source_image, result, json.dumps(metadata, ensure_ascii=False, indent=2)

    with gr.Blocks(css=css, title="Emoji Diffusion Editor", theme=gr.themes.Soft()) as demo:
        gr.HTML(
            """
            <div class="hero">
              <h1>Emoji Diffusion Editor</h1>
              <p>Upload an emoji or choose one from the curated dataset, describe the edit in natural language,
              and generate a refined emoji result with diffusion-based image editing.</p>
            </div>
            """
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=4, elem_classes=["panel"]):
                source_mode = gr.Radio(choices=["Choose built-in emoji", "Upload"], value="Choose built-in emoji", label="Source Mode")
                vendor_dropdown = gr.Dropdown(choices=vendors, value=default_vendor, label="Vendor Style", visible=True)
                emoji_dropdown = gr.Dropdown(choices=default_choices, value=default_key, label="Emoji Selection", visible=True)
                upload_image = gr.Image(label="Upload Emoji", type="pil", image_mode="RGBA", visible=False)
                preview_image = gr.Image(value=load_preview(default_entry.image_path), label="Source Preview", type="pil", interactive=False)
                selection_text = gr.Markdown(f"Selected: {default_entry.display_name} in {default_entry.vendor} style", elem_classes=["meta-box"])

            with gr.Column(scale=5, elem_classes=["panel"]):
                instruction_box = gr.Textbox(
                    label="Edit Instruction",
                    lines=4,
                    placeholder="例如：把这个 emoji 变得更伤心一点，并保留苹果风格的立体阴影。",
                )
                extra_style_hint = gr.Textbox(
                    label="Optional Style Hint",
                    lines=2,
                    placeholder="例如：keep the clean Apple emoji shading and centered face composition",
                )
                with gr.Accordion("Advanced Controls", open=False):
                    negative_prompt = gr.Textbox(label="Negative Prompt", value=DEFAULT_NEGATIVE_PROMPT, lines=3)
                    with gr.Row():
                        steps = gr.Slider(10, 80, value=30, step=1, label="Inference Steps")
                        resolution = gr.Slider(128, 512, value=256, step=32, label="Resolution")
                    with gr.Row():
                        guidance_scale = gr.Slider(1.0, 12.0, value=4.5, step=0.1, label="Text Guidance")
                        image_guidance_scale = gr.Slider(1.0, 5.0, value=1.8, step=0.1, label="Image Guidance")
                    with gr.Row():
                        seed = gr.Number(value=-1, precision=0, label="Seed (-1 for random)")
                        scheduler_name = gr.Dropdown(choices=[("Euler A", "euler_a"), ("DPM++", "dpm")], value="euler_a", label="Scheduler")
                run_button = gr.Button("Edit Emoji", variant="primary")

            with gr.Column(scale=6, elem_classes=["panel"]):
                with gr.Row():
                    source_out = gr.Image(label="Resolved Source", type="pil", interactive=False)
                    result_out = gr.Image(label="Edited Result", type="pil", interactive=False)
                metadata_out = gr.Code(label="Run Metadata", language="json")

        gr.Examples(
            examples=[
                ["把这个 emoji 变得更开心一点，并保留整体平台风格。"],
                ["Change this emoji into a crying version with visible tears."],
                ["Render the same expression in a cleaner Google-style emoji design."],
                ["Add sunglasses and make the face feel more confident."],
            ],
            inputs=[instruction_box],
        )

        source_mode.change(
            fn=lambda mode: (
                gr.update(visible=mode == "Choose built-in emoji"),
                gr.update(visible=mode == "Choose built-in emoji"),
                gr.update(visible=mode == "Upload"),
            ),
            inputs=[source_mode],
            outputs=[vendor_dropdown, emoji_dropdown, upload_image],
        )
        vendor_dropdown.change(fn=update_emoji_choices, inputs=[vendor_dropdown], outputs=[emoji_dropdown, preview_image, selection_text])
        emoji_dropdown.change(fn=update_preview, inputs=[emoji_dropdown], outputs=[preview_image, selection_text])
        run_button.click(
            fn=run_edit,
            inputs=[
                source_mode,
                upload_image,
                emoji_dropdown,
                instruction_box,
                negative_prompt,
                steps,
                guidance_scale,
                image_guidance_scale,
                seed,
                resolution,
                extra_style_hint,
                scheduler_name,
            ],
            outputs=[source_out, result_out, metadata_out],
        )
    return demo


def main() -> int:
    args = parse_args()
    demo = build_interface(args)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
