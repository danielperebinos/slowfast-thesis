# Dashboard: SlowFast + YOLO Live Inference — Phase 2 Implementation Plan

## Context

Phase 1 trained 4 SlowFast variants (Baseline, Attention, ROI, Hybrid) with action anticipation loss
on the AVA dataset. The models and YOLOv8 (`experiments/yolov8n.pt`) were already present in the
project. Phase 2 builds an interactive dashboard to run these models on a video file (or webcam
when available), displaying person detections, predicted actions, and performance metrics in
real time.

**UI framework chosen:** Gradio — minimal code, built-in streaming support, ideal for ML demos.

---

## Architecture

```
dashboard/
├── app.py                    # Gradio app entry point
├── inference/
│   ├── __init__.py
│   ├── pipeline.py           # InferencePipeline: orchestrates YOLO + SlowFast
│   ├── buffer.py             # FrameBuffer: circular deque (32 frames)
│   └── model_loader.py       # load_model(): checkpoint → experiment model
├── labels/
│   ├── kinetics400.txt        # 400 Kinetics action labels (pretrained fallback)
│   └── ava_actions.txt        # 80 AVA action labels (custom-trained models)
└── videos/
    └── *.mp4                  # Drop video files here for simulation
```

### Data Flow

```
Video File (dashboard/videos/)
    │
    ▼
FrameBuffer (deque, maxlen=32)
    │
    ├── Every frame:
    │   └── YOLOv8 person detection → bounding boxes for visualization
    │
    └── Every K frames (configurable, default=8):
        └── PackPathway (reused from experiments/core/data_loader.py)
            ├── Slow path: uniformly subsample 4 frames from buffer
            └── Fast path: all 32 frames
                └── SlowFast forward → softmax → Top-3 actions
                    └── Annotate frame → Gradio output image
```

---

## Implementation Plan

### Step 1: Dependencies

Added to `pyproject.toml`:
- `gradio>=4.0` (installed: 6.8.0)
- `opencv-python>=4.8` (installed: 4.11.0)

### Step 2: `dashboard/inference/buffer.py` ✅

`FrameBuffer` class:
- `deque(maxlen=32)` stores raw BGR frames
- `add(frame)`: append frame
- `is_ready()`: `len(deque) == 32`
- `get_pathways()`: returns `(slow_tensor, fast_tensor)` with shape `(1, C, T, H, W)`
  - Resize short side to 256, center crop to 224×224
  - Normalize: mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]
  - Fast: all 32 frames → `(1, 3, 32, 224, 224)`
  - Slow: uniformly subsample 4 frames → `(1, 3, 4, 224, 224)`
  - Uses `PackPathway` from `experiments/core/data_loader.py`

### Step 3: `dashboard/inference/model_loader.py` ✅

`load_model(variant, checkpoint_dir, device)`:
- Variant → model class mapping:
  - `"baseline"` → `create_slowfast()` from pytorchvideo
  - `"attention"` → `AttentionSlowFast` from `experiments/experiment_02/model.py`
  - `"roi"` → `RoiGuidanceSlowFast` from `experiments/experiment_03/model.py`
  - `"hybrid"` → `HybridSlowFast` from `experiments/experiment_04/model.py`
- Checkpoint search: looks for `<variant>` in filename under `checkpoint_dir`, or
  `<checkpoint_dir>/<variant>/checkpoint.pth`; reads `num_classes` from checkpoint dict
- Fallback: loads pretrained SlowFast-R50 from pytorchvideo hub (Kinetics-400, 400 classes)
- Returns `(model, num_classes, label_type)` where `label_type` is `"kinetics"` or `"ava"`

### Step 4: `dashboard/inference/pipeline.py` ✅

`InferencePipeline` class:

```
__init__(variant, checkpoint_dir, device)
    - loads model via model_loader
    - loads YOLOv8 from experiments/yolov8n.pt
    - creates FrameBuffer(maxlen=32)
    - state: frame_count, last_predictions, last_latency_ms, rolling FPS window

process_frame(frame_rgb, threshold, inference_interval)
    → (annotated_frame_rgb, predictions, metrics_dict)

    1. BGR conversion → YOLO inference (filter class=0 "person", conf >= threshold)
    2. buffer.add(frame_bgr)
    3. if buffer.is_ready() and frame_count % inference_interval == 0:
         slow, fast = buffer.get_pathways()
         t0 = perf_counter()
         logits = model([slow, fast])
         last_latency_ms = (perf_counter() - t0) * 1000
         probs = softmax(logits[0])
         top3 = [(labels[i], conf) for i, conf in topk(probs, 3)]
    4. annotate: YOLO boxes + "person" label + action text overlay (top-left)
       + FPS / latency text (bottom-left)
    5. rolling 30-frame FPS average
    6. return annotated_rgb, last_predictions, {fps, latency_ms, inf_per_sec}
```

### Step 5: `dashboard/labels/` ✅

- `kinetics400.txt`: 400 lines, one Kinetics-400 action per line
- `ava_actions.txt`: 80 AVA action labels

### Step 6: `dashboard/app.py` ✅

Gradio `Blocks`-based UI:

```
Row 1 (controls):
  - model_dropdown: ["Baseline", "Attention", "ROI", "Hybrid"]
  - threshold_slider: 0.1–0.9, default=0.5
  - interval_slider: 1–32 (inference every N frames), default=8
  - load_btn: "Load Model" → triggers model loading

Row 2 (video I/O):
  - left column:
      - gr.Video(sources=["upload"]) — upload video file
      - gr.Dropdown — quick-pick from dashboard/videos/ (shown when files exist)
      - "Simulate" button → triggers generator
      - "Stop" button → cancels generator
  - right column:
      - gr.Image — annotated output frames

Row 3 (predictions + metrics):
  - gr.Dataframe(headers=["Action", "Confidence"]) — Top-3 predictions
  - gr.JSON — {fps, latency_ms, inf_per_sec}
```

**Simulation loop** (generator, no webcam needed):
```python
cap = cv2.VideoCapture(video_path)
while True:
    ret, frame = cap.read()
    annotated, preds, metrics = pipeline.process_frame(frame_rgb, ...)
    yield annotated, pred_table, metrics, state
    time.sleep(max(0, 1/fps_target - elapsed))  # pace to original FPS
```

### Step 7: Model hot-swap ✅

When user changes model dropdown and clicks "Load Model":
- Old pipeline is garbage-collected (Python ref-counting)
- New `InferencePipeline(variant, checkpoint_dir, device)` is created
- New pipeline stored in `gr.State`

---

## Critical Files

| File | Role |
|---|---|
| `dashboard/app.py` | Gradio UI + simulation loop |
| `dashboard/inference/pipeline.py` | YOLO + SlowFast orchestration |
| `dashboard/inference/buffer.py` | Sliding window frame buffer |
| `dashboard/inference/model_loader.py` | Checkpoint loading for all 4 variants |
| `experiments/core/data_loader.py` | **Reused** `PackPathway` + transforms |
| `experiments/experiment_02/model.py` | **Reused** `AttentionSlowFast` |
| `experiments/experiment_03/model.py` | **Reused** `RoiGuidanceSlowFast` |
| `experiments/experiment_04/model.py` | **Reused** `HybridSlowFast` |
| `experiments/yolov8n.pt` | **Reused** existing YOLO weights |
| `pyproject.toml` | Added `gradio`, `opencv-python` |

---

## Verification

```bash
uv run python dashboard/app.py
# → http://localhost:7860
```

1. Select "Baseline", click **Load Model** (downloads pretrained weights if no checkpoint found)
2. Drop a `.mp4` into `dashboard/videos/` and pick it from the dropdown (or upload directly)
3. Click **Simulate**
4. Verify:
   - YOLO bounding boxes appear around persons
   - Action labels update every N frames
   - FPS counter and latency (ms) are displayed
   - Switching model variant and clicking **Load Model** reloads inference correctly

---

## Implementation Notes

- `gradio 6.x` removed the `mirror_webcam` kwarg from `gr.Image` — not used
- `gr.Video` is used as input (not `gr.Image` streaming) since no physical camera is available
- The simulation generator paces output to match the source video's native FPS
- YOLO path is resolved relative to `experiments/yolov8n.pt` (not project root)
- Checkpoint keys: `model_state_dict` (training scripts) with `num_classes` stored in the dict
- `label_type = "ava"` when a checkpoint is found; `"kinetics"` (400 classes) for pretrained fallback
