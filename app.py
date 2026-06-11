"""Interactive Gradio UI for the emoji diffusion editor."""

from __future__ import annotations

import json
import base64
from io import BytesIO
from dataclasses import dataclass
from html import escape
from pathlib import Path

import gradio as gr
from PIL import Image

from emoji_editing.catalog import EmojiCatalogEntry
from emoji_editing.diffusion_inference import choices_for_vendor, edit_emoji_image, load_ui_catalog
from emoji_editing.prompting import DEFAULT_NEGATIVE_PROMPT


@dataclass(frozen=True)
class UIConfig:
    base_model: str
    lora_path: Path
    vendor_index_csv: Path
    precision: str
    device: str | None
    host: str
    port: int
    share: bool


# 在这里修改可视化界面配置。
UI_CONFIG = UIConfig(
    base_model="timbrooks/instruct-pix2pix",  # 推理底座模型。
    lora_path=Path("artifacts/emoji_diffusion_editor_60k/lora_final"),  # 训练完成后的 LoRA 目录。
    vendor_index_csv=Path("data/interim/emoji_editing/metadata/vendor_image_index.csv"),  # 内置 emoji 索引表。
    precision="fp16",  # RTX 单卡界面推理通常先用 fp16。
    device="cuda:0",  # 强制使用第一张 GPU。
    host="127.0.0.1",  # Gradio 监听地址。
    port=7860,  # Gradio 端口。
    share=False,  # 是否生成公网分享链接。
)


EXAMPLE_PROMPTS = [
    "把这个 emoji 变得更开心一点，并保留整体平台风格。",
    "Change this emoji into a crying version with visible tears.",
    "Render the same expression in a cleaner Google-style emoji design.",
    "Add sunglasses and make the face feel more confident.",
]


def build_theme() -> gr.themes.Base:
    return gr.themes.Base(
        primary_hue=gr.themes.colors.teal,
        secondary_hue=gr.themes.colors.orange,
        neutral_hue=gr.themes.colors.stone,
    ).set(
        body_background_fill="#f4f1eb",
        body_background_fill_dark="#f4f1eb",
        body_text_color="#1d1b18",
        body_text_color_dark="#1d1b18",
        body_text_color_subdued="#6c665d",
        body_text_color_subdued_dark="#6c665d",
        background_fill_primary="#ffffff",
        background_fill_primary_dark="#ffffff",
        background_fill_secondary="#f8f6f1",
        background_fill_secondary_dark="#f8f6f1",
        block_background_fill="#ffffff",
        block_background_fill_dark="#ffffff",
        block_border_color="#ded8ce",
        block_border_color_dark="#ded8ce",
        block_border_width="1px",
        block_radius="8px",
        block_shadow="none",
        block_shadow_dark="none",
        block_label_background_fill="transparent",
        block_label_background_fill_dark="transparent",
        block_label_text_color="#6c665d",
        block_label_text_color_dark="#6c665d",
        block_title_text_color="#1d1b18",
        block_title_text_color_dark="#1d1b18",
        input_background_fill="#fbfaf7",
        input_background_fill_dark="#fbfaf7",
        input_background_fill_focus="#ffffff",
        input_background_fill_focus_dark="#ffffff",
        input_border_color="#d8d1c7",
        input_border_color_dark="#d8d1c7",
        input_border_color_focus="#0f766e",
        input_border_color_focus_dark="#0f766e",
        input_radius="7px",
        button_primary_background_fill="#111111",
        button_primary_background_fill_dark="#111111",
        button_primary_background_fill_hover="#0f766e",
        button_primary_background_fill_hover_dark="#0f766e",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        button_primary_border_color="#111111",
        button_primary_border_color_dark="#111111",
        button_primary_shadow="none",
        button_secondary_background_fill="#ffffff",
        button_secondary_background_fill_dark="#ffffff",
        button_secondary_background_fill_hover="#f0eee8",
        button_secondary_background_fill_hover_dark="#f0eee8",
        button_secondary_text_color="#1d1b18",
        button_secondary_text_color_dark="#1d1b18",
        button_secondary_border_color="#d8d1c7",
        button_secondary_border_color_dark="#d8d1c7",
        button_large_radius="8px",
        button_medium_radius="7px",
        button_small_radius="6px",
        slider_color="#0f766e",
        slider_color_dark="#0f766e",
        accordion_text_color="#1d1b18",
        accordion_text_color_dark="#1d1b18",
    )


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --page: #f7f7f4;
  --surface: #ffffff;
  --surface-soft: #f1f2ed;
  --ink: #181817;
  --muted: #686a61;
  --line: #dedfd7;
  --teal: #0f766e;
  --gold: #c7862e;
  --coral: #c95c46;
}

.gradio-container {
  max-width: 1320px !important;
  margin: 0 auto !important;
  padding: 18px 24px 24px !important;
  color: var(--ink) !important;
  background: var(--page) !important;
  font-family: "Inter", ui-sans-serif, system-ui, sans-serif !important;
}

.gradio-container .contain {
  gap: 14px !important;
}

footer {
  display: none !important;
}

.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 18px;
  margin-bottom: 16px;
  padding: 4px 2px 14px;
}

.brand-lockup {
  display: flex;
  align-items: center;
  gap: 12px;
}

.brand-mark {
  width: 40px;
  height: 40px;
  display: grid;
  place-items: center;
  border-radius: 10px;
  background: linear-gradient(135deg, #ffe08a, #f6b45b);
  color: #211a0c;
  font-size: 22px;
  box-shadow: inset 0 0 0 1px rgba(24, 24, 23, 0.12);
}

.brand-copy h1 {
  margin: 0;
  color: var(--ink);
  font-size: 1.42rem;
  line-height: 1;
  font-weight: 800;
  letter-spacing: 0;
}

.brand-copy p {
  margin: 5px 0 0;
  color: var(--muted);
  font-size: 0.9rem;
}

.run-spec {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 10px;
}

.spec-pill {
  display: inline-flex;
  align-items: center;
  min-height: 30px;
  padding: 0 11px;
  border: 1px solid var(--line);
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.82);
  color: var(--muted);
  font-family: "JetBrains Mono", ui-monospace, monospace;
  font-size: 0.72rem;
  white-space: nowrap;
}

.workspace {
  align-items: flex-start !important;
  gap: 16px !important;
}

.control-panel,
.stage-panel {
  border: 1px solid var(--line) !important;
  border-radius: 10px !important;
  background: var(--surface) !important;
  box-shadow: 0 18px 48px rgba(24, 24, 23, 0.07) !important;
}

.control-panel {
  padding: 18px !important;
}

.stage-panel {
  padding: 0 !important;
  overflow: hidden !important;
}

.section-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin: 0 0 10px;
  color: var(--ink);
  font-weight: 700;
  font-size: 0.86rem;
}

.section-title .index {
  color: var(--teal);
  font-family: "JetBrains Mono", ui-monospace, monospace;
  font-size: 0.74rem;
  font-weight: 500;
}

.divider {
  height: 1px;
  margin: 14px 0;
  background: var(--line);
}

.selected-meta {
  min-height: 34px;
  margin-top: 8px;
  padding: 8px 10px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--surface-soft);
  color: var(--muted);
  font-size: 0.84rem;
}

.output-stage {
  padding: 18px;
  min-height: 506px;
  background:
    radial-gradient(circle at 20% 0%, rgba(15, 118, 110, 0.10), transparent 34%),
    radial-gradient(circle at 100% 12%, rgba(199, 134, 46, 0.13), transparent 32%),
    #fbfbf7;
}

.output-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 18px;
  padding-bottom: 14px;
  border-bottom: 1px solid rgba(24, 24, 23, 0.09);
}

.output-header h2 {
  margin: 0;
  font-size: 1.18rem;
  line-height: 1;
  font-weight: 800;
  letter-spacing: 0;
}

.output-header span {
  color: var(--muted);
  font-family: "JetBrains Mono", ui-monospace, monospace;
  font-size: 0.72rem;
}

.comparison-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 18px;
}

.image-card {
  border: 1px solid rgba(24, 24, 23, 0.10);
  border-radius: 10px;
  background: #ffffff;
  overflow: hidden;
  box-shadow: 0 12px 34px rgba(24, 24, 23, 0.06);
}

.image-card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 14px;
  border-bottom: 1px solid rgba(24, 24, 23, 0.08);
}

.image-card-title {
  color: var(--ink);
  font-size: 0.78rem;
  font-weight: 750;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.image-card-kicker {
  color: var(--muted);
  font-family: "JetBrains Mono", ui-monospace, monospace;
  font-size: 0.68rem;
}

.image-canvas {
  min-height: 332px;
  display: grid;
  place-items: center;
  padding: 26px;
  background:
    linear-gradient(45deg, rgba(24,24,23,0.035) 25%, transparent 25%),
    linear-gradient(-45deg, rgba(24,24,23,0.035) 25%, transparent 25%),
    linear-gradient(45deg, transparent 75%, rgba(24,24,23,0.035) 75%),
    linear-gradient(-45deg, transparent 75%, rgba(24,24,23,0.035) 75%),
    #f7f7f3;
  background-size: 22px 22px;
  background-position: 0 0, 0 11px, 11px -11px, -11px 0;
}

.image-canvas img {
  width: min(100%, 244px);
  height: min(244px, 52vh);
  object-fit: contain;
  image-rendering: auto;
}

.empty-result {
  width: min(100%, 244px);
  aspect-ratio: 1;
  display: grid;
  place-items: center;
  border: 1px dashed rgba(24, 24, 23, 0.18);
  border-radius: 10px;
  color: #8a877d;
  background: rgba(255, 255, 255, 0.58);
  font-family: "JetBrains Mono", ui-monospace, monospace;
  font-size: 0.76rem;
}

.source-note {
  margin-top: 14px;
  padding: 11px 12px;
  border: 1px solid rgba(24, 24, 23, 0.08);
  border-radius: 9px;
  background: rgba(255, 255, 255, 0.76);
  color: var(--muted);
  font-size: 0.84rem;
}

.gradio-container [data-testid="image"],
.gradio-container .image-container {
  overflow: hidden !important;
  border: 1px solid rgba(29, 27, 24, 0.12) !important;
  border-radius: 9px !important;
  background: #f8f6f1 !important;
}

.gradio-container textarea,
.gradio-container input,
.gradio-container select {
  font-size: 0.94rem !important;
}

.gradio-container label > span {
  font-size: 0.78rem !important;
  font-weight: 600 !important;
  color: var(--muted) !important;
}

.gradio-container .form,
.gradio-container .block,
.gradio-container .gr-group {
  box-shadow: none !important;
}

.gradio-container .gr-accordion,
.gradio-container details {
  border: 1px solid var(--line) !important;
  border-radius: 9px !important;
  background: var(--surface-soft) !important;
}

#run-edit-btn {
  width: 100% !important;
  min-height: 48px !important;
  margin-top: 14px !important;
  font-size: 0.98rem !important;
  font-weight: 750 !important;
  letter-spacing: 0 !important;
  border-radius: 8px !important;
}

#run-edit-btn:hover {
  transform: translateY(-1px);
}

@media (max-width: 900px) {
  .gradio-container {
    padding: 12px !important;
  }
  .topbar {
    align-items: flex-start;
    flex-direction: column;
  }
  .run-spec {
    justify-content: flex-start;
  }
  .control-panel,
  .stage-panel {
    padding: 14px !important;
  }
  .output-stage {
    min-height: auto;
    padding: 14px;
  }
  .comparison-grid {
    grid-template-columns: 1fr;
  }
  .image-canvas { min-height: 300px; }
}
"""


def build_interface(config: UIConfig) -> gr.Blocks:
    catalog_bundle = load_ui_catalog(config.vendor_index_csv)
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

    def image_to_data_uri(image: Image.Image) -> str:
        buffer = BytesIO()
        image.convert("RGBA").save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def render_stage_html(
        source_image: Image.Image | None,
        edited_image: Image.Image | None = None,
        source_caption: str = "Ready",
        edited_caption: str = "Waiting",
    ) -> str:
        source_body = (
            f'<img src="{image_to_data_uri(source_image)}" alt="Source emoji">'
            if source_image is not None
            else '<div class="empty-result">Source</div>'
        )
        edited_body = (
            f'<img src="{image_to_data_uri(edited_image)}" alt="Edited emoji">'
            if edited_image is not None
            else '<div class="empty-result">Edited result</div>'
        )
        return f"""
        <div class="output-stage">
          <div class="output-header">
            <h2>Output</h2>
            <span>before / after</span>
          </div>
          <div class="comparison-grid">
            <div class="image-card">
              <div class="image-card-header">
                <span class="image-card-title">Source</span>
                <span class="image-card-kicker">{escape(source_caption)}</span>
              </div>
              <div class="image-canvas">{source_body}</div>
            </div>
            <div class="image-card">
              <div class="image-card-header">
                <span class="image-card-title">Edited</span>
                <span class="image-card-kicker">{escape(edited_caption)}</span>
              </div>
              <div class="image-canvas">{edited_body}</div>
            </div>
          </div>
        </div>
        """

    def selected_label(entry: EmojiCatalogEntry) -> str:
        return f"Selected: {entry.display_name} · {entry.vendor}"

    def update_emoji_choices(vendor: str):
        vendor_choices = choices_for_vendor(entries, vendor)
        selected_key = vendor_choices[0][1]
        entry = lookup[selected_key]
        preview = load_preview(entry.image_path)
        return (
            gr.Dropdown(choices=vendor_choices, value=selected_key),
            selected_label(entry),
            render_stage_html(preview, source_caption=entry.vendor),
        )

    def update_preview(selected_key: str):
        entry = lookup[selected_key]
        preview = load_preview(entry.image_path)
        return selected_label(entry), render_stage_html(preview, source_caption=entry.vendor)

    def toggle_source_mode(source_mode: str, selected_key: str, uploaded_image: Image.Image | None):
        built_in = source_mode == "Choose built-in emoji"
        if built_in:
            entry = lookup[selected_key]
            source_image = load_preview(entry.image_path)
            stage_html = render_stage_html(source_image, source_caption=entry.vendor)
        else:
            stage_html = render_stage_html(uploaded_image, source_caption="Upload")
        return (
            gr.update(visible=built_in),
            gr.update(visible=built_in),
            gr.update(visible=not built_in),
            stage_html,
        )

    def update_uploaded_stage(uploaded_image: Image.Image | None) -> str:
        return render_stage_html(uploaded_image, source_caption="Upload")

    def run_edit(
        source_mode: str,
        uploaded_image: Image.Image | None,
        selected_key: str,
        instruction: str | None,
        negative_prompt: str | None,
        steps: int,
        guidance_scale: float,
        image_guidance_scale: float,
        seed: int,
        resolution: int,
        extra_style_hint: str | None,
        scheduler_name: str,
    ):
        instruction = (instruction or "").strip()
        extra_style_hint = (extra_style_hint or "").strip()
        negative_prompt = negative_prompt or DEFAULT_NEGATIVE_PROMPT

        if not instruction:
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
            base_model=config.base_model,
            lora_path=config.lora_path if config.lora_path.exists() else None,
            precision=config.precision,
            device=config.device,
            source_name=source_name,
            source_vendor=source_vendor,
            steps=steps,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
            negative_prompt=negative_prompt,
            seed=int(seed),
            resolution=int(resolution),
            scheduler_name=scheduler_name,
            extra_style_hint=extra_style_hint or None,
        )
        _ = json.dumps(metadata, ensure_ascii=False, indent=2)
        return render_stage_html(source_image, result, source_caption="Input", edited_caption=f"seed {metadata['seed']}")

    with gr.Blocks(title="Emoji Diffusion Editor") as demo:
        gr.HTML(
            """
            <div class="topbar">
              <div class="brand-lockup">
                <div class="brand-mark">🙂</div>
                <div class="brand-copy">
                  <h1>Emoji Diffusion Editor</h1>
                  <p>Natural-language emoji editing studio</p>
                </div>
              </div>
              <div class="run-spec">
                <span class="spec-pill">InstructPix2Pix</span>
                <span class="spec-pill">LoRA 60K</span>
                <span class="spec-pill">11 styles</span>
              </div>
            </div>
            """
        )

        with gr.Row(equal_height=False, elem_classes=["workspace"]):
            with gr.Column(scale=4, min_width=340, elem_classes=["control-panel"]):
                gr.HTML('<div class="section-title"><span>Source</span><span class="index">01</span></div>')
                source_mode = gr.Radio(
                    choices=["Choose built-in emoji", "Upload"],
                    value="Choose built-in emoji",
                    label="Mode",
                )
                with gr.Row():
                    vendor_dropdown = gr.Dropdown(choices=vendors, value=default_vendor, label="Vendor", visible=True)
                    emoji_dropdown = gr.Dropdown(choices=default_choices, value=default_key, label="Emoji", visible=True)
                upload_image = gr.Image(label="Upload", type="pil", image_mode="RGBA", visible=False, height=150)
                selection_text = gr.Markdown(selected_label(default_entry), elem_classes=["selected-meta"])

                gr.HTML('<div class="divider"></div>')
                gr.HTML('<div class="section-title"><span>Instruction</span><span class="index">02</span></div>')
                instruction_box = gr.Textbox(
                    label="Prompt",
                    lines=3,
                    placeholder="例如：把这个 emoji 变得更伤心一点，并保留苹果风格的立体阴影。",
                )
                run_button = gr.Button("Generate Edit", variant="primary", elem_id="run-edit-btn")
                example_picker = gr.Dropdown(
                    choices=EXAMPLE_PROMPTS,
                    value=None,
                    label="Examples",
                )

                with gr.Accordion("Advanced", open=False):
                    extra_style_hint = gr.Textbox(
                        label="Style Hint",
                        lines=2,
                        placeholder="keep the clean Apple emoji shading and centered face composition",
                    )
                    negative_prompt = gr.Textbox(label="Negative Prompt", value=DEFAULT_NEGATIVE_PROMPT, lines=2)
                    with gr.Row():
                        steps = gr.Slider(10, 80, value=40, step=1, label="Steps")
                        resolution = gr.Slider(128, 512, value=256, step=32, label="Resolution")
                    with gr.Row():
                        guidance_scale = gr.Slider(1.0, 12.0, value=3.5, step=0.1, label="Text")
                        image_guidance_scale = gr.Slider(1.0, 5.0, value=2.8, step=0.1, label="Image")
                    with gr.Row():
                        seed = gr.Number(value=-1, precision=0, label="Seed")
                        scheduler_name = gr.Dropdown(
                            choices=[("Euler A", "euler_a"), ("DPM++", "dpm")],
                            value="dpm",
                            label="Scheduler",
                        )

            with gr.Column(scale=7, min_width=520, elem_classes=["stage-panel"]):
                stage_view = gr.HTML(render_stage_html(load_preview(default_entry.image_path), source_caption=default_entry.vendor))

        example_picker.change(fn=lambda choice: choice or "", inputs=[example_picker], outputs=[instruction_box])
        source_mode.change(
            fn=toggle_source_mode,
            inputs=[source_mode, emoji_dropdown, upload_image],
            outputs=[vendor_dropdown, emoji_dropdown, upload_image, stage_view],
        )
        vendor_dropdown.change(
            fn=update_emoji_choices,
            inputs=[vendor_dropdown],
            outputs=[emoji_dropdown, selection_text, stage_view],
        )
        emoji_dropdown.change(
            fn=update_preview,
            inputs=[emoji_dropdown],
            outputs=[selection_text, stage_view],
        )
        upload_image.change(fn=update_uploaded_stage, inputs=[upload_image], outputs=[stage_view])
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
            outputs=[stage_view],
        )
    return demo


def main() -> int:
    config = UI_CONFIG
    demo = build_interface(config)
    demo.launch(
        server_name=config.host,
        server_port=config.port,
        share=config.share,
        theme=build_theme(),
        css=CUSTOM_CSS,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
