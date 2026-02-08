# Experiments Registry (SlowFast Anticipation)

| ID | Name | Architecture | Technical Objective |
| :--- | :--- | :--- | :--- |
| **01** | Baseline | SlowFast (R50) | Baseline for mAP & TTA. |
| **02** | Attention | SlowFast + Non-Local | Evaluate global correlations via Self-Attention. |
| **03** | ROI Guidance | SlowFast + YOLOv8 | Reduce redundancy by masking background on Fast Pathway. |
| **04** | Hybrid | ROI + Local Attention | Pareto optimization: high accuracy at minimal latency. |

## Core Logic (`core/`)
- **`loss.py`**: $L_{total} = L_{cls} + \alpha \cdot \exp\left(\frac{t - T_{start}}{T}\right)$
- **`metrics.py`**: mAP, Top-k Acc, TTA (Time-To-Action).
- **`data_loader.py`**: Standardized ingestion of (Video, Label) pairs.

## How to Run
Each experiment is autonomous.

```bash
# Baseline
python experiment_01/train.py

# Attention
python experiment_02/train.py

# ROI
python experiment_03/train.py

# Hybrid
python experiment_04/train.py
```

All runs are logged to **MLflow**.
