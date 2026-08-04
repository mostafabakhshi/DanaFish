# DanaFish

Automated pipeline for counting neuronal cell bodies in zebrafish spinal-cord images: orientation correction, YOLO-based neuron detection, spinal-cord region analysis, and Excel export.

---

## Table of Contents
- [Requirements](#requirements)
- [Environment Setup](#environment-setup)
- [Running the Pipeline](#running-the-pipeline)
- [Pipeline Stages](#pipeline-stages)
- [Data Structure](#data-structure)
- [Output Structure](#output-structure)
- [Model Weights](#model-weights)
- [Evaluation](#evaluation)

---

## Requirements

- Python 3.10+
- CUDA-compatible GPU (recommended; CPU works but is slow)
- Conda (Anaconda or Miniconda)

---

## Environment Setup

The environment `zebrafish310` already exists on the primary workstation. To recreate it elsewhere:

```bash
conda create -n zebrafish310 python=3.10 -y
conda activate zebrafish310
pip install -r requirements.txt
```

> For GPU support (CUDA 11.8):
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
> ```

---

## Running the Pipeline

### Important: run from the project root

`config.py` resolves the model weights by **relative** path, so the pipeline only works when the
current directory is the project root:

```powershell
cd D:\Project2\zebraFish\v10Y0726
```

Running from anywhere else fails at model load.

### Step 1 — Select the environment

```powershell
conda activate zebrafish310
```

If this returns `CondaError: Run 'conda init' before 'conda activate'`, conda is not hooked into
PowerShell. Either run `conda init powershell` once and open a new terminal, or skip activation
entirely and call the environment's interpreter directly:

```powershell
C:\Users\mstfb\anaconda3\envs\zebrafish310\python.exe main.py
```

### Step 2 — Run

```powershell
python main.py
```

`main.py` is interactive and prompts for two paths:

```
Enter the path to your folder containing images:              <- e.g. PLXNA1-Jesh-JPG-Final
Enter the path for results folder (press Enter for default):  <- Enter = datasets/zebra_yolo/test/My_Prediction/results
```

The input folder is walked **recursively**, so pointing it at a dataset root processes every
subfolder in one run. The output tree mirrors the input tree.

### Non-interactive alternatives

Pipe the two answers in:

```powershell
"PLXNA1-Jesh-JPG-Final`nresults_plxna1" | python main.py
```

Or call the entry point directly, which is cleaner for scripted/batch runs:

```powershell
python -c "from main import process_images; process_images('PLXNA1-Jesh-JPG-Final', 'results_plxna1')"
```

---

## Pipeline Stages

Each image passes through four stages (`main.py::process_images`):

1. **Orientation correction** — `ImageRotationCorrector` (`image_rotation_corrector.py`) detects
   anatomical landmarks with `rotation_model_v1` (4-class: body, head, neuron, tail), reorients the
   larva to a standard pose, and returns an 840x840 canvas plus the body bounding box.
   Images with no detected body region are skipped with a warning.
2. **Neuron detection** — `ZebraFishModel` (`model.py`) runs `neuron_v7_yolo26m_1280` on the
   corrected image at `imgsz=1280`, `conf=0.30`, `iou=0.50`.
3. **Region analysis** — `ExactBodyRegionAnalyzer` (`test_exact_body_region_pipeline.py`) restricts
   detections to the spinal-cord region, fits the spinal-cord curve, and groups neurons into
   segments via DBSCAN on the x-coordinate.
4. **Export** — annotated images and per-image + summary spreadsheets.

---

## Data Structure

Any nested folder layout works; `main.py` walks the tree recursively and accepts `.jpg`, `.jpeg`,
and `.png`. The two datasets currently in the repo:

```
PLXNA1-Jesh-JPG-Final/          (60 images)
├── 190823/
│   ├── Ctrl/
│   └── plxna1 sb/
├── 190911/
│   ├── Ctrl/
│   └── plxna1a sb/
└── 190922/
    ├── Ctrl/
    └── plxna1 sb/

Thabiso_MP_Fish-Images/         (98 images)
├── nTiO2 Pure/
│   ├── 72 HPF/
│   ├── 96 HPF/
│   └── 120 HPF/
├── nZnO Pure/
└── ...                         (COL 1, CRE 3, SOA 1, SUN 1 — photo-transformation groups)
```

---

## Output Structure

The output tree mirrors the input tree:

```
<output_folder>/
├── <subfolder>/                        ← mirrors input structure
│   ├── annotated_<image>.jpg           ← neurons + fitted spinal-cord line
│   ├── metrics_<image_stem>.xlsx       ← per-image segment measurements
│   └── regions/
│       └── regions_<image>.jpg         ← original (left) | corrected + landmarks (right)
└── summary.xlsx                        ← one row per image, at the output root
```

`summary.xlsx` columns: `Image`, `Neurons`, `Line_px` (spinal-cord length in pixels),
`Rotation_deg`, `Annotated`, `Regions`.

Note that `metrics_<stem>.xlsx` is written only when the analyzer produced segment data; images
where no neurons fall in the spinal-cord region get an annotated image but no spreadsheet.

---

## Model Weights

Both are tracked in `runs/` and loaded by relative path:

| Model | Path | Purpose |
|---|---|---|
| `rotation_model_v1` | `runs/detect/rotation_model_v1/weights/best.pt` | Landmark detection for orientation correction (4-class) |
| `neuron_v7_yolo26m_1280` | `runs/detect/runs/detect/neuron_v7_yolo26m_1280/weights/best.pt` | Neuron detection (YOLO26m, mAP@50 = 0.907) |

The neuron weights path is doubly nested (`runs/detect/runs/detect/`) — this is intentional and
matches `NEURON_MODEL_PATH` in `config.py`.

---

## Evaluation

`eval_v7.py` evaluates the neuron detector against manual annotations on the held-out test set and
produces agreement statistics (Pearson r, MAE, Bland–Altman bias and limits of agreement) plus a
confidence sweep.

```powershell
python eval_v7.py
```

Inference is whole-image at `imgsz=1280`, the same configuration the production pipeline uses and
the one reported in the manuscript.
