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

## Semi-Automatic Mode

The project provides two entry points that share the same models and the same analysis:

- `main.py` — fully automatic, processes a folder unattended.
- `manual.py` — semi-automatic, for correcting the model where it gets an image wrong.

`manual.py` requires PyQt5 (included in `requirements.txt`); `main.py` does not.

```powershell
python manual.py
```

`main.py` is unchanged — the fully automatic path still behaves exactly as before.

### Workflow

**Open an image and the whole pipeline runs immediately** — orientation correction, region
detection, neuron detection and spinal-cord fitting. You are presented with a finished result and
only intervene where it is wrong.

The panel offers two modes, **Edit** (the default, since correcting the result is the job) and
**View** for panning around without changing anything. Edit fans out into what you are correcting,
and lands on **Neurons** after each image loads:

| Edit target | What you do |
|---|---|
| **Head** `H` / **Tail** `T` | Drag to place the landmark when the detector was wrong or found nothing; `Delete` erases it. Then `Ctrl+R` re-applies the orientation from your corrected landmarks. |
| **Region of interest** `R` | Drag to redraw the region. Drag a corner handle to adjust, `Delete` to clear. `Ctrl+D` re-detects inside it. |
| **Spinal cord** `S` | Drag a control point to **bend** the curve · click empty space to **insert** one · right-click or `Delete` to **remove** one · `Ctrl+Shift+R` refits automatically. |
| **Neurons** `N` | Drag empty space to **add** · right-click to **delete** · drag a box to **move** · drag a corner to **resize**. The list has a tick box per neuron with **Remove ticked**, plus **Clear all neurons** to start counting by hand. |

Then `Ctrl+S` exports the annotated image, Excel metrics, an edit log, and appends to
`summary_manual.xlsx`.

> **`body` is not shown.** The detector's body box is still used internally — it sets the
> orientation and seeds the region of interest — but it is neither drawn nor offered as an edit
> target, because it nearly coincides with the ROI and having two near-identical rectangles on
> screen is just noise.

### Editing the spinal cord

The cord is a curve, so it is edited as a **smooth curve through a small number of control
points** rather than drawn freehand — a hand-drawn line is jittery and hard to place precisely,
whereas a spline through ~10 handles is both easier and closer to the real anatomy.

When detection runs, the automatic fit is sampled into 10 draggable handles. Drag one to correct a
region, click to insert a handle where you need finer control, right-click to remove one. Below
four handles the curve degrades gracefully (quadratic, then linear) rather than failing.

An edited cord is not cosmetic — it is passed back into the analysis, so it drives which neurons
are kept, the distance-to-curve values, and the reported length. Verified on a test image: dragging
the cord below the neurons drops the count from 12 to 0, and a deliberately wavy cord changes both
the count (8) and the length (625 px vs 453 px).

Re-running detection preserves your cord edits; `Ctrl+Shift+R` discards them and refits.

### Units

Lengths are shown in **pixels by default**. The side panel has an explicit Pixels / Micrometres
choice; the µm option becomes selectable once a scale is set (`Ctrl+K`), so the interface never
implies a physical measurement it cannot make.

The choice affects the display only. If a calibration exists, exported results always carry the µm
columns regardless of which unit is on screen — switching the view to pixels never strips
measurements from your saved data.

Keys: `V` view · `E` edit · `H` head · `T` tail · `R` region · `S` cord · `N` neurons ·
`M` measure · `Ctrl+K` set scale · `Ctrl+Z` undo · `Ctrl+0` fit view ·
`PgUp`/`PgDn` previous/next image · `F1` help. Mouse wheel zooms; middle-drag pans.

Neurons are colour-coded by origin — **green** came from the model, **gold** was placed by hand —
so it stays visible how much of a count is automated and how much is manual.

### Measuring in micrometres

Press `Ctrl+K` to calibrate. Either type in µm-per-pixel directly, or press `M`, drag along a
distance you already know (a scale bar in the image, or a feature of known size), then reopen
`Ctrl+K` and enter what that distance actually is — the dialog converts your measurement into a
calibration. The value is written to `calibration.json` and reloaded in later sessions.

Once calibrated:

- a microscopy-style **scale bar** appears bottom-right, snapped to a round length,
- the **ruler** ticks along the top and left edges switch from pixels to µm,
- the **ROI label** and side panel report width × height in µm live as you drag,
- the **spinal-cord length** is reported in µm,
- the exported spreadsheet gains `Line_um`, `ROI_width_um`, `ROI_height_um`,
  `Neurons_per_100um` and `um_per_px`,
- the exported annotated image has the scale bar **burned in**, so figures are publication-ready.

Uncalibrated, everything falls back to pixels and nothing else changes.

> **Why calibration is stored per *original* pixel.** The pipeline fits every image onto an
> 840 × 840 canvas by `scale_factor = min(840/w, 840/h)`, and that factor differs per image — a
> 2056 × 685 micrograph is displayed at 0.409×. Storing µm-per-*displayed*-pixel would therefore be
> wrong for every image of a different size. The calibration is a property of the objective and
> camera, so it is held in original-image pixels and converted per image as
> `µm_per_displayed_px = µm_per_original_px / scale_factor`. On a real 2056 × 685 image, ignoring
> this would under-report lengths by **2.45×**.

### Output

Alongside the `annotated_*.jpg` and `metrics_*.xlsx` that the automatic pipeline produces,
semi-automatic mode writes two extra files for provenance:

- `edits_<stem>.json` — the ROI, every neuron with its origin, and every neuron deleted by hand.
- `summary_manual.xlsx` — one row per image with `Neurons`, `From_model`, `Added_by_hand`,
  `Deleted_by_hand`, `Line_px`, `ROI`, `Confidence`, and which landmarks were edited.

Re-exporting an image replaces its existing row rather than duplicating it.

### One deliberate difference from the automatic pipeline

`ExactBodyRegionAnalyzer.analyze_exact_body_region` discards neurons whose centre sits on or above
the fitted spinal-cord line. That is correct for automatic runs but would silently delete a neuron
the operator had just placed by hand. The method therefore takes an optional `protected_boxes`
argument, which `manual.py` populates with the hand-placed neurons so they always survive.
It defaults to `None`, leaving `main.py`'s behaviour bit-for-bit identical.

---

## Pipeline Stages

Each image passes through five stages (`main.py::process_images`):

1. **Preprocessing** — `preprocessing.py` removes the background pedestal (see below). Applied
   inside `correct_image_orientation`, before either model runs, so both see the same pixels.
2. **Orientation correction** — `ImageRotationCorrector` (`image_rotation_corrector.py`) detects
   anatomical landmarks with the active landmark detector (4-class: body, head, neuron, tail),
   reorients the larva to a standard pose, and returns an 840x840 canvas plus the body bounding
   box. Images with no detected body region are skipped with a warning.
3. **Neuron detection** — `ZebraFishModel` (`model.py`) runs `neuron_v7_yolo26m_1280` on the
   corrected image at `imgsz=1280`, `conf=0.35`, `iou=0.50`.
4. **Region analysis** — `ExactBodyRegionAnalyzer` (`test_exact_body_region_pipeline.py`) restricts
   detections to the spinal-cord region, fits the spinal-cord curve, and groups neurons into
   segments via DBSCAN on the x-coordinate.
5. **Export** — annotated images and per-image + summary spreadsheets.

### Step 1 — background pedestal removal

Some acquisition sessions carry a constant additive offset across the frame, so that true black
sits at intensity 25–35 instead of near zero. The offset compresses the range available to the
signal and depresses detection confidence; in the worst cases genuine cell bodies fall below the
detection threshold and the image is counted as zero neurons.

The step measures the offset from the image itself — the 1st percentile of non-zero green-channel
pixels — and subtracts it. Nothing is hard-coded, so the correction scales to each acquisition, and
because the value comes from the image's own histogram it is blind to experimental group.

It is **gated**: images whose pedestal is below `min_pedestal` (15) are passed through untouched.
Applied unconditionally the step also acts on images that do not have the problem, where it
over-detects — on the held-out test set an ungated correction raised bias from +0.17 to +0.92
counts per image. Gated, 51 of those 52 images are left unchanged (MAE 1.52 → 1.60,
bias +0.17 → +0.25).

```python
PREPROCESS_CONFIG = {"enabled": True, "percentile": 1.0, "channel": 1, "min_pedestal": 15.0}
```

Setting `enabled: False` restores the previous behaviour exactly — verified to reproduce
pre-existing counts on all 42 images of the biological dataset.

> **A caveat worth stating.** The annotated splits contain almost no pedestal-affected images
> (median pedestal 10, versus 26 in the affected acquisition), so they cannot measure the benefit,
> and what they do measure is within noise — the validation split marginally prefers the correction
> (r 0.816 → 0.855 ungated) while the test split marginally prefers it off (r 0.862 → 0.844). The
> justification for the step is the failure mode it removes, not a gain in benchmark accuracy.

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

Both detectors are YOLO26m, tracked in `runs/` and loaded by relative path:

| Model | Path | Purpose |
|---|---|---|
| `neuron_v7_yolo26m_1280` | `runs/detect/runs/detect/neuron_v7_yolo26m_1280/weights/best.pt` | Neuron detection (1-class) |
| `landmark_v8_yolo26m_union` | `runs/detect/runs/detect/landmark_v8_yolo26m_union/weights/best.pt` | Landmark detection (4-class: b, h, n, t) |

Both were trained from `yolo26m.pt` at `imgsz=1280`, batch 2, AdamW, seed 42, capped at 300 epochs
with patience 50. The paths are doubly nested (`runs/detect/runs/detect/`) — intentional, and matches
`config.py`.

| | neuron detector | landmark detector |
|---|---|---|
| dataset | `fish13_v8_neuron_only` | `fish13_union_4class` |
| split | 424 / 62 / 52 | 700 / 99 / 65 |
| instances | 10,011 | 18,685 |
| ran to / best epoch | 149 / 111 | 263 / 225 |
| test mAP@50 | 0.906 | 0.943 |
| test mAP@50-95 | 0.394 | 0.824 |

### The landmark dataset is pooled across acquisitions

`fish13_union_4class/data.yaml` lists two annotation sets rather than copying them:
`fish13.v4-bnhtsize.yolov11` (276/37/13) and `fish13.v8-danafish28072026.yolo26` (424/62/52). They
share no images and both declare `nc: 4` with `names: ['b','h','n','t']` in the same order, so the
labels pool directly.

Pooling is not incidental — it is required. A landmark detector trained on either session alone
generalizes poorly to the other, and in particular the **body/spine class** fails to transfer, which
matters because that box seeds the spinal-cord region. Trained on the pooled set, the detector
reaches mAP@50 0.943 overall (body 0.969, head 0.990, tail 0.995) on its own held-out test set and
locates a body in all 42 larvae of the biological dataset.

Neither annotation set contains any plxna1a image, so the biological dataset is fully held out from
training.

### Multi-scale body search

The body/spine class is sensitive to the larva's apparent scale, which varies with frame size and
body curvature. The detector therefore searches for the body over the inference sizes in
`LANDMARK_CONFIG["body_search_scales"]` — `(960, 416, 320)` — taking the first detection at or above
`body_search_accept` (0.1) and otherwise the most confident found. Head and tail are read from the
same pass as the accepted body, so all landmarks stay mutually consistent.

Single-scale inference locates a body in 40 of the 42 biological larvae; the three-scale search
locates one in all 42.

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
