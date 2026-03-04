"""
SlowFast + YOLO Live Inference Dashboard (Phase 2).

Run:
    uv run python dashboard/app.py
    # or from project root:
    python -m dashboard.app

Drop any .mp4 / .avi / .mov file into dashboard/videos/ then pick it in the UI.
"""

import sys
from pathlib import Path

import cv2
import gradio as gr
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard.inference.pipeline import InferencePipeline

_CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "volumes"
_VIDEOS_DIR = Path(__file__).resolve().parent / "videos"

VARIANT_MAP = {
    "Baseline": "baseline",
    "Attention": "attention",
    "ROI": "roi",
    "Hybrid": "hybrid",
}

_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

_STATUS_READY = (
    "<span style='display:inline-flex;align-items:center;gap:6px;font-weight:600'>"
    "<span style='width:10px;height:10px;border-radius:50%;background:#22c55e;display:inline-block'></span>"
    "{label} &nbsp;READY</span>"
)
_STATUS_ERROR = (
    "<span style='display:inline-flex;align-items:center;gap:6px;font-weight:600;color:#ef4444'>"
    "<span style='width:10px;height:10px;border-radius:50%;background:#ef4444;display:inline-block'></span>"
    "ERROR</span>"
)
_STATUS_IDLE = (
    "<span style='display:inline-flex;align-items:center;gap:6px;color:#9ca3af'>"
    "<span style='width:10px;height:10px;border-radius:50%;background:#9ca3af;display:inline-block'></span>"
    "No model loaded</span>"
)


def _render_predictions_html(predictions: list) -> str:
    if not predictions:
        return "<p style='color:#888; font-size:0.85rem'>Waiting for inference…</p>"
    rows = ""
    for label, conf in predictions:
        pct = int(conf * 100)
        rows += (
            f"<div style='margin:4px 0; font-size:0.85rem'>"
            f"<span>{label}</span>"
            f"<span style='float:right; color:#555'>{conf:.2f}</span>"
            f"<div style='clear:both; background:#e5e7eb; border-radius:4px; height:6px; margin-top:3px'>"
            f"<div style='background:#6366f1; width:{pct}%; height:100%; border-radius:4px'></div>"
            f"</div></div>"
        )
    return rows


# ── Model loading ───────────────────────────────────────────────────────────────

def load_pipeline(variant_display: str, state: dict) -> tuple[str, dict]:
    """Load (or hot-swap) the InferencePipeline."""
    variant = VARIANT_MAP[variant_display]
    try:
        logger.info("Loading pipeline for variant '{}'", variant_display)
        pipeline = InferencePipeline(variant=variant, checkpoint_dir=_CHECKPOINT_DIR)
        state["pipeline"] = pipeline
        logger.success("Model '{}' loaded successfully.", variant_display)
        return _STATUS_READY.format(label=variant_display), state
    except Exception as exc:
        logger.exception("Failed to load model '{}'", variant_display)
        return _STATUS_ERROR, state


# ── Video file simulation ───────────────────────────────────────────────────────

def simulate_video(video_path: str, threshold: float, interval: int, state: dict):
    """
    Generator: reads a video file frame by frame, runs the pipeline,
    and yields (annotated_frame, pred_html, fps_val, lat_val, buf_val, state).
    """
    if video_path is None:
        yield None, _render_predictions_html([]), 0.0, 0.0, 0.0, state
        return

    pipeline: InferencePipeline | None = state.get("pipeline")
    if pipeline is None:
        yield None, "<p style='color:#ef4444'>No model loaded. Click <b>Load Model</b> first.</p>", 0.0, 0.0, 0.0, state
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Cannot open video: {}", video_path)
        yield None, f"<p style='color:#ef4444'>Cannot open video: {video_path}</p>", 0.0, 0.0, 0.0, state
        return

    logger.info("Starting simulation: {}", Path(video_path).name)

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            annotated, predictions, metrics = pipeline.process_frame(
                frame_rgb=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
                confidence_threshold=threshold,
                inference_interval=int(interval),
            )

            yield (
                annotated,
                _render_predictions_html(predictions),
                float(metrics["fps"]),
                float(metrics["latency_ms"]),
                float(metrics["buffer_fill"]),
                state,
            )
    finally:
        cap.release()
        logger.info("Simulation ended: {}", Path(video_path).name)


# ── Gradio UI ───────────────────────────────────────────────────────────────────

def _list_videos() -> list[str]:
    if not _VIDEOS_DIR.exists():
        return []
    return sorted(str(p) for p in _VIDEOS_DIR.iterdir() if p.suffix.lower() in _VIDEO_EXTS)


def build_ui() -> gr.Blocks:
    local_videos = _list_videos()

    with gr.Blocks(theme=gr.themes.Soft(), title="SlowFast Live Dashboard") as demo:
        gr.HTML(
            "<h2 style='margin:0 0 4px'>⚡ SlowFast + YOLO Dashboard</h2>"
            "<p style='color:#6b7280;margin:0'>Real-time action recognition &amp; person detection</p>"
        )

        state = gr.State({"pipeline": None})

        with gr.Row():
            # ── Sidebar ───────────────────────────────────────────────────────
            with gr.Column(scale=1, min_width=220):
                gr.Markdown("### Model")
                model_dropdown = gr.Dropdown(
                    choices=list(VARIANT_MAP.keys()),
                    value="Baseline",
                    label="Variant",
                    container=False,
                )
                load_btn = gr.Button("Load Model", variant="primary")
                status_html = gr.HTML(_STATUS_IDLE)

                with gr.Accordion("Settings", open=False):
                    threshold_slider = gr.Slider(
                        minimum=0.1, maximum=0.9, step=0.05, value=0.5,
                        label="Detection Confidence",
                    )
                    interval_slider = gr.Slider(
                        minimum=1, maximum=32, step=1, value=8,
                        label="Inference Every N Frames",
                    )

                gr.Markdown("### Metrics")
                fps_num = gr.Number(label="FPS", precision=1, interactive=False, value=0.0)
                lat_num = gr.Number(label="Latency ms", precision=0, interactive=False, value=0.0)
                buf_num = gr.Number(label="Buffer (frames)", precision=0, interactive=False, value=0.0)

                gr.Markdown("### Predictions")
                pred_html = gr.HTML(_render_predictions_html([]))

            # ── Main panel ───────────────────────────────────────────────────
            with gr.Column(scale=3):
                with gr.Row():
                    video_input = gr.Video(
                        label="Input Video",
                        sources=["upload"],
                    )
                    output_feed = gr.Image(type="numpy", label="Annotated Output")

                with gr.Row():
                    if local_videos:
                        video_picker = gr.Dropdown(
                            choices=local_videos,
                            label="Or pick from dashboard/videos/",
                            value=None,
                        )
                        video_picker.change(fn=lambda p: p, inputs=video_picker, outputs=video_input)

                with gr.Row():
                    simulate_btn = gr.Button("▶ Simulate", variant="primary")
                    stop_btn = gr.Button("⏹ Stop")

        # ── Event handlers ────────────────────────────────────────────────────
        load_btn.click(
            fn=load_pipeline,
            inputs=[model_dropdown, state],
            outputs=[status_html, state],
        )

        sim_event = simulate_btn.click(
            fn=simulate_video,
            inputs=[video_input, threshold_slider, interval_slider, state],
            outputs=[output_feed, pred_html, fps_num, lat_num, buf_num, state],
        )

        stop_btn.click(fn=None, cancels=[sim_event])

    return demo


if __name__ == "__main__":
    logger.info("Starting SlowFast dashboard")
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
