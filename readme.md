# PROIECT TEHNIC (CLI AGENT) — Optimizarea Arhitecturii SlowFast pentru Anticipare (Faza 1)

## 0. Scop
Construirea unei soluții de viziune computerizată capabile să **recunoască acțiuni umane online** și să **anticipeze** evenimente cât mai devreme, prin **reducerea redundanței spațio-temporale**. Proiectul urmărește un **compromis Pareto** între:
- **Acuratețe** (Top-1 / mAP)
- **Timp de anticipare** (Time-to-Action / TTA) + latență / throughput

> Faza 2 (aplicație real-time) este doar schițată la final și nu este implementată în acest scope.

---

## 1. Obiective (Faza 1 — Cercetare & Antrenare)
CLI agentul trebuie să livreze:

1) **3 variante arhitecturale** + baseline:
- **Baseline:** SlowFast standard (ResNet-50).
- **Varianta A — Attention (Global / Non-Local):** module Self-Attention inserate în blocuri superioare (ex: `res_4`/`res_5`).
- **Varianta B — ROI Guidance (YOLOv8):** mască ROI din YOLOv8 + produs Hadamard pe intrarea/feature-urile căii rapide.
- **Varianta C — Hybrid:** atenție aplicată **doar** pe tokeni/features din ROI (complexitate redusă).

2) **Funcție de pierdere cu penalizare de latență decizională** (asimetrie pentru “decizie târzie”):
- Formă recomandată (conform descrierii inițiale):
  \[
  L_{total} = L_{cls} + \alpha \cdot \exp\left(\frac{t - T_{start}}{T}\right)
  \]
  unde `t` = timpul/offset-ul curent în fereastra temporală, `T_start` = pragul după care penalizarea crește, `T` = scală, `α` = greutate.
- Alternativ (pentru implementare clară):
  \[
  L_{total} = L_{cls} + \alpha \cdot L_{lat}
  \]
  cu `L_lat` definit explicit ca penalizare monotonic crescătoare în timp.

3) **Evaluare comparativă** pe benchmark-uri:
- **Kinetics-400** (pre-antrenare backbone / diversitate)
- **AVA** (ROI + adnotări bounding box la nivel de cadru; mAP)
- (Opțional pentru analiză temporală fină) **Something-Something V2**

4) **Raport final** cu:
- metrici (Top-1, mAP, TTA, FPS, FP/min)
- comparație A/B/C vs baseline
- selecția variantei candidate pentru Faza 2

---

## 2. Definiții & Metrici
### 2.1 Metrici obligatorii
- **Top-1 Accuracy** (clasificare)
- **mAP** (în special pe AVA)
- **Time-to-Action (TTA)**  
  Definiție de implementat: timpul (secunde) dintre **prima predicție validă** (peste prag, stabilă) și **momentul de început vizibil al acțiunii** / timestamp ground-truth.
- **Throughput (FPS)** în inferență (batch controlat)
- **False Positives per Minute** (zone fără activitate relevantă)

### 2.2 Reguli de fairness (pentru benchmarking corect)
Toate experimentele trebuie să folosească:
- același subset/split (când se compară direct)
- același pipeline de preprocesare
- aceleași seed-uri (reproductibilitate)
- MLflow tracking obligatoriu (config + metrici + artefacte)

---

## 3. Dataset & Ingerare
### 3.1 Structură recomandată
- clipuri stocate comprimat (optimizare I/O)
- fișiere de index (CSV/JSON) cu:
  - `path`, `label`, `start_time`, `end_time`, (și pentru AVA: `boxes`, `frame_timestamps`)

### 3.2 Data pipeline (PyTorchVideo)
- tensori video: \((C, T, H, W)\)
- augmentări obligatorii:
  - **Random Short Side Scale**
  - **Temporal Jittering** (robustețe la variații de viteză)
  - (opțional) Center Crop / Random Crop

---

## 4. Arhitecturi de implementat
### 4.1 Baseline — SlowFast (ResNet-50)
- configurare sampling:
  - **Slow pathway:** 4 cadre, stride 16 (\(\tau = 16\))
  - **Fast pathway:** 32 cadre, stride 2 (\(\alpha = 8\), \(\beta = 1/8\))
- loss: Cross-Entropy standard (fără latență)

### 4.2 Varianta A — Attention (Global)
- inserare module Non-Local / Self-Attention în `res_4` (și opțional `res_5`)
- logică:
  - din tensorul \(X \in \mathbb{R}^{C \times T \times H \times W}\) se formează proiecții \(Q, K, V\) și se calculează afinitatea globală
- ipoteză: crește mAP pentru acțiuni complexe, dar costă latență (ms/frame)

### 4.3 Varianta B — ROI Guidance cu YOLOv8
- rulează **YOLOv8** pe cadre (sincron sau asincron, dar raportat)
- generează mască binară \(M\) din bounding boxes
- aplică produs Hadamard pe calea rapidă:
  \[
  \hat{X}_{fast} = X_{fast} \odot M
  \]
- loss: `L_total = L_cls + α L_lat` (α explorat)
- obiectiv: eliminarea fundalului static ⇒ alertă mai timpurie (TTA mai bun)

### 4.4 Varianta C — Hybrid (ROI + Attention Localizată)
- atenție aplicată doar pe tokenii/features din ROI
- eficiență:
  - atenția se reduce de la \(O(N^2)\) la \(O(N \cdot N_{ROI})\)
- țintă: viteză apropiată de B, precizie apropiată de A

---

## 5. Protocolul Experimentelor (Faza 1)
> Toate run-urile sunt track-uite în MLflow (params, metrics, artefacte, checkpoint-uri).

### Experiment 1 — Baseline
- model: SlowFast ResNet-50
- loss: CE
- output: referință pentru Top-1, mAP, TTA, FPS

### Experiment 2 — Varianta A (Attention)
- model: baseline + Non-Local/Self-Attention (`res_4`, opțional `res_5`)
- metrică cheie: câștig mAP vs cost latență (ms/frame)

### Experiment 3 — Varianta B (YOLO ROI)
- model: baseline + YOLO ROI mask + Hadamard pe fast pathway
- sweep α (ex: `[0.0, 0.1, 0.25, 0.5, 1.0]` sau configurabil)
- metrică cheie: TTA îmbunătățit + FP/min controlat

### Experiment 4 — Varianta C (Hybrid)
- model: ROI + atenție localizată
- țintă: stabilitate alertă timpurie + menținerea throughput-ului

---

## 6. Antrenare & Optimizare
### 6.1 Optimizatori
- AdamW **sau** SGD cu momentum
- weight decay: \(10^{-4}\) (configurabil)

### 6.2 Schedules
- Warm-up (primele 10 epoci) + **Cosine Annealing LR**

### 6.3 Early Stopping
- pe mAP (AVA) sau metrică principală definită per dataset
- salvează cel mai bun checkpoint + ultimul checkpoint

---

## 7. MLflow: Cerințe de tracking
Trebuie logate minim:
- Params: dataset version, split id, seed, model variant, α, LR, batch size, num epochs, augmentări, threshold-uri
- Metrics: loss train/val, Top-1, mAP, TTA, FPS, FP/min
- Artefacte: config yaml/json, grafice curbe, (opțional) Grad-CAM vizualizări, checkpoint-uri

---

## 8. Analiză calitativă (obligatoriu pentru B/C)
- generează **Grad-CAM / activation maps** pe sample-uri fixe
- verifică dacă ROI Guidance focalizează corect pe zone active
- extrage “failure cases” (cazuri limită) pentru raport

---

## 9. Livrabile (Faza 1)
1) Cod complet + configurații pentru: baseline, A, B, C  
2) CLI care rulează:
   - ingest / index
   - train
   - evaluate
   - export rezultate (CSV/JSON)
   - vizualizări (opțional)
3) MLflow runs complete (comparabile)
4) Raport final (markdown/pdf) cu concluzia: varianta optimă pentru Faza 2

---

## 10. CLI Contract (interfață minimă)
CLI-ul trebuie să fie reproductibil și scriptabil.

### 10.1 Comenzi recomandate
- `slowfast-cli data index --dataset <name> --input <path> --out <index.json>`
- `slowfast-cli train --variant {baseline,attn,roi,hybrid} --dataset <name> --config <cfg.yaml>`
- `slowfast-cli eval --run-id <mlflow_run_id> --split val --out metrics.json`
- `slowfast-cli benchmark --variants baseline,attn,roi,hybrid --out results.csv`
- `slowfast-cli viz gradcam --run-id <id> --samples <list> --out <dir>`

### 10.2 Ieșiri standard
- `results.csv` (tabel comparativ)
- `metrics.json` (pe run)
- `artifacts/` (vizualizări, curbe, exemple)
- checkpoint-uri: `checkpoints/<variant>/<run_id>/`

---

## 11. Mediu & Reproductibilitate
### 11.1 Docker (obligatoriu)
- imagine bazată pe `nvidia/cuda`
- suport NVIDIA Container Toolkit
- rulează pe Debian/Ubuntu
- include: PyTorch, PyTorchVideo, Ultralytics YOLOv8, MLflow

### 11.2 Reguli de reproducere
- seed fix (torch, numpy, random)
- log versioning: git commit hash + dataset version
- un singur “source of truth” pentru config (yaml)

---

## 12. Tehnologii (rezumat)
- **Framework AI:** PyTorch, PyTorchVideo, YOLOv8 (Ultralytics)
- **Experiment Tracking:** MLflow
- **Infrastructură:** Docker, NVIDIA CUDA Toolkit
- **Analiză date:** Jupyter Lab, Pandas, Matplotlib

---

## 13. Faza 2 (Real-Time) — doar schiță (NU se implementează acum)
- RTSP ingestion (hardware decode)
- circular buffer (ferestre glisante)
- thresholding + notificări / alerte

---

## 14. Criterii de acceptare (Definition of Done)
- Pot rula **4 experimente** (baseline + A/B/C) din CLI, end-to-end.
- Rezultatele apar în MLflow cu parametri și metrici complete.
- Există un `results.csv` comparativ + concluzie (varianta aleasă) bazată pe Pareto (precizie vs TTA vs FPS).
- Codul este modular (variantă selectabilă fără duplicare masivă).
