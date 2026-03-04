"""
SlowFast + YOLO Live Inference Dashboard (Phase 2).

Run:
    python dashboard/app.py
    # or from project root:
    python -m dashboard.app

Drop any .mp4 / .avi / .mov file into dashboard/videos/ then upload it in the UI.
"""

import sys
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
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


# ── Model loading ──────────────────────────────────────────────────────────────

def load_pipeline(variant_display: str, state: dict) -> tuple[str, dict]:
    """Load (or hot-swap) the InferencePipeline."""
    variant = VARIANT_MAP[variant_display]
    try:
        logger.info("Loading pipeline for variant '{}'", variant_display)
        pipeline = InferencePipeline(variant=variant, checkpoint_dir=_CHECKPOINT_DIR)
        state["pipeline"] = pipeline
        msg = f"Model '{variant_display}' loaded successfully."
        logger.success(msg)
        return msg, state
    except Exception as exc:
        logger.exception("Failed to load model '{}'", variant_display)
        return f"Error loading model: {exc}", state


# ── Video file simulation ──────────────────────────────────────────────────────

def simulate_video(video_path: str, threshold: float, interval: int, state: dict):
    """
    Generator: reads a video file frame by frame, runs the pipeline,
    and yields (annotated_frame, pred_table, metrics, state) for Gradio.
    """
    if video_path is None:
        yield None, [], {"error": "No video file selected."}, state
        return

    pipeline: InferencePipeline | None = state.get("pipeline")
    if pipeline is None:
        yield None, [], {"error": "No model loaded. Click 'Load Model' first."}, state
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Cannot open video: {}", video_path)
        yield None, [], {"error": f"Cannot open video: {video_path}"}, state
        return

    fps_target = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_delay = 1.0 / fps_target
    logger.info("Starting simulation: {} @ {:.1f} fps", Path(video_path).name, fps_target)

    try:
        while True:
            t0 = time.perf_counter()
            ret, frame_bgr = cap.read()
            if not ret:
                break

            annotated, predictions, metrics = pipeline.process_frame(
                frame_rgb=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
                confidence_threshold=threshold,
                inference_interval=int(interval),
            )
            pred_table = [[label, f"{conf:.3f}"] for label, conf in predictions]

            sleep_time = frame_delay - (time.perf_counter() - t0)
            if sleep_time > 0:
                time.sleep(sleep_time)

            yield annotated, pred_table, metrics, state
    finally:
        cap.release()
        logger.info("Simulation ended: {}", Path(video_path).name)


# ── Gradio UI ──────────────────────────────────────────────────────────────────

def _list_videos() -> list[str]:
    if not _VIDEOS_DIR.exists():
        return []
    return sorted(str(p) for p in _VIDEOS_DIR.iterdir() if p.suffix.lower() in _VIDEO_EXTS)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="SlowFast Live Dashboard") as demo:
        gr.Markdown("# SlowFast + YOLO Live Inference Dashboard")
        gr.Markdown(
            "1. Select a model variant and click **Load Model**.\n"
            "2. Upload a video file (or pick one from `dashboard/videos/`).\n"
            "3. Click **Simulate** to run inference frame by frame."
        )

        state = gr.State({"pipeline": None})

        # ── Row 1: controls ────────────────────────────────────────────
        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=list(VARIANT_MAP.keys()),
                value="Baseline",
                label="Model Variant",
            )
            threshold_slider = gr.Slider(
                minimum=0.1, maximum=0.9, step=0.05, value=0.5,
                label="Detection Confidence Threshold",
            )
            interval_slider = gr.Slider(
                minimum=1, maximum=32, step=1, value=8,
                label="Inference Every N Frames",
            )
            load_btn = gr.Button("Load Model", variant="primary")

        status_text = gr.Textbox(label="Status", interactive=False, value="No model loaded.")

        # ── Row 2: video input + output ────────────────────────────────
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(
                    label="Input Video (upload or pick from dashboard/videos/)",
                    sources=["upload"],
                )
                local_videos = _list_videos()
                if local_videos:
                    video_picker = gr.Dropdown(
                        choices=local_videos,
                        label="Or select from dashboard/videos/",
                        value=None,
                    )
                    video_picker.change(fn=lambda p: p, inputs=video_picker, outputs=video_input)

                simulate_btn = gr.Button("Simulate", variant="primary")
                stop_btn = gr.Button("Stop")

            output_feed = gr.Image(type="numpy", label="Annotated Output")

        # ── Row 3: predictions + metrics ──────────────────────────────
        with gr.Row():
            pred_table = gr.Dataframe(
                headers=["Action", "Confidence"],
                datatype=["str", "str"],
                label="Top-3 Predictions",
                row_count=(3, "fixed"),
            )
            metrics_json = gr.JSON(label="Metrics (fps / latency_ms / inf_per_sec)")

        # ── Event handlers ─────────────────────────────────────────────
        load_btn.click(
            fn=load_pipeline,
            inputs=[model_dropdown, state],
            outputs=[status_text, state],
        )

        sim_event = simulate_btn.click(
            fn=simulate_video,
            inputs=[video_input, threshold_slider, interval_slider, state],
            outputs=[output_feed, pred_table, metrics_json, state],
        )

        stop_btn.click(fn=None, cancels=[sim_event])

    return demo


if __name__ == "__main__":
    logger.info("Starting SlowFast dashboard")
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
