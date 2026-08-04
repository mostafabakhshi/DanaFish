"""
Post-training evaluation for neuron model v7.
Dataset: fish13_v8_neuron_only  (52 test images, 2048x1708 px, full resolution)
Model:   YOLO26m — neuron_v7_yolo26m_1280/weights/best.pt

Inference is whole-image at imgsz=1280, matching the production pipeline.
"""
import os, glob, shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats
from ultralytics import YOLO

ROOT       = Path(r"D:\Project2\zebraFish\v88tU1")
MODEL_PATH = ROOT / "runs/detect/runs/detect/neuron_v7_yolo26m_1280/weights/best.pt"
DATA_YAML  = ROOT / "datasets/fish13_v8_neuron_only/data.yaml"
TEST_IMGS  = ROOT / "datasets/fish13_v8_neuron_only/test/images"
TEST_LBLS  = ROOT / "datasets/fish13_v8_neuron_only/test/labels"
PAPER      = Path(r"D:\Project2\DanaFishPaperFinal")

CONF = 0.30
IOU  = 0.5
IMGSZ = 1280

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

img_files = sorted(glob.glob(str(TEST_IMGS / "*.jpg")) +
                   glob.glob(str(TEST_IMGS / "*.png")))
gt = []
for p in img_files:
    lbl = TEST_LBLS / f"{Path(p).stem}.txt"
    gt.append(sum(1 for ln in open(lbl) if ln.strip()) if lbl.exists() else 0)
gt = np.array(gt, dtype=float)
print(f"Test set: {len(img_files)} images  GT counts: min={int(gt.min())} max={int(gt.max())} mean={gt.mean():.1f}")

# ── 1. model.val() ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  1. model.val() on v7 test set")
print("="*60)
model = YOLO(str(MODEL_PATH))
val_results = model.val(
    data=str(DATA_YAML), imgsz=IMGSZ, conf=0.001, iou=0.6,
    split="test", project=str(ROOT / "runs/detect"),
    name="val_v7_testset", exist_ok=True, workers=0,
)
map50   = val_results.box.map50
map5095 = val_results.box.map
mp      = val_results.box.mp
mr      = val_results.box.mr
f1      = 2 * mp * mr / (mp + mr + 1e-9)
print(f"\n  mAP@50       : {map50:.4f}")
print(f"  mAP@50-95    : {map5095:.4f}")
print(f"  Precision    : {mp:.4f}")
print(f"  Recall       : {mr:.4f}")
print(f"  F1-score     : {f1:.4f}")

val_dir = ROOT / "runs/detect/val_v7_testset"
for name in ("BoxPR_curve.png", "confusion_matrix_normalized.png"):
    src = val_dir / name
    if src.exists():
        shutil.copy(src, PAPER / name)
        print(f"  Copied {name} → paper folder")

# ── 2. Counting agreement (whole-image inference) ─────────────────────────────
print("\n" + "="*60)
print(f"  2. Whole-image inference (conf={CONF}, imgsz={IMGSZ})")
print("="*60)
preds = []
for p in img_files:
    res = model(p, conf=CONF, iou=IOU, imgsz=IMGSZ, verbose=False)
    preds.append(len(res[0].boxes) if res[0].boxes is not None else 0)
preds = np.array(preds, dtype=float)

diff   = preds - gt
r_, p_ = stats.pearsonr(gt, preds)
mae    = np.mean(np.abs(diff))
bias   = np.mean(diff)
sd     = np.std(diff, ddof=1)
loa_lo = bias - 1.96 * sd
loa_hi = bias + 1.96 * sd
print(f"  Pearson r : {r_:.4f}  (p={p_:.2e})")
print(f"  MAE       : {mae:.2f}")
print(f"  Bias      : {bias:+.4f}")
print(f"  SD        : {sd:.4f}")
print(f"  95% LoA   : [{loa_lo:.2f}, {loa_hi:.2f}]")

# ── 3. Scatter + Bland-Altman ─────────────────────────────────────────────────
mean_pair = (preds + gt) / 2
slope, intercept, *_ = stats.linregress(gt, preds)

fig = plt.figure(figsize=(14, 6))
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

ax1 = fig.add_subplot(gs[0])
ax1.scatter(gt, preds, color="#2196F3", s=60, zorder=3, label=f"Images (n={len(img_files)})")
lim = max(gt.max(), preds.max()) + 3
ax1.plot([0, lim], [0, lim], "k--", lw=1, label="Identity")
x_fit = np.linspace(0, lim, 100)
ax1.plot(x_fit, slope * x_fit + intercept, "r-", lw=1.5,
         label=f"Fit  y={slope:.2f}x+{intercept:.2f}")
ax1.set_xlabel("Manual count (ground truth)", fontsize=11)
ax1.set_ylabel("Automated count (YOLO26m)", fontsize=11)
ax1.set_title(f"Automated vs Manual\nPearson r = {r_:.3f}  (p < 0.001)",
              fontsize=12, fontweight="bold")
ax1.legend(fontsize=9); ax1.set_xlim(0, lim); ax1.set_ylim(0, lim)
ax1.text(0.05, 0.92, f"MAE = {mae:.2f}  n = {len(img_files)}", transform=ax1.transAxes,
         fontsize=9, color="gray")

ax2 = fig.add_subplot(gs[1])
ax2.scatter(mean_pair, diff, color="#4CAF50", s=60, zorder=3)
ax2.axhline(bias,   color="blue", lw=1.5, linestyle="-",  label=f"Bias  {bias:+.2f}")
ax2.axhline(loa_hi, color="red",  lw=1.2, linestyle="--", label=f"+1.96 SD  {loa_hi:.2f}")
ax2.axhline(loa_lo, color="red",  lw=1.2, linestyle="--", label=f"−1.96 SD  {loa_lo:.2f}")
ax2.axhline(0, color="black", lw=0.8, linestyle=":")
ax2.set_xlabel("Mean of automated & manual", fontsize=11)
ax2.set_ylabel("Automated − Manual  (difference)", fontsize=11)
ax2.set_title("Bland-Altman Plot", fontsize=12, fontweight="bold")
ax2.legend(fontsize=9)

fig.suptitle("YOLO26m Neuron Detector v7: Validation Against Manual Annotations\n"
             f"({len(img_files)}-image independent test set, whole-image conf={CONF})",
             fontsize=12, fontweight="bold", y=1.02)
out_val = PAPER / "validation_03.png"
plt.savefig(str(out_val), dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved {out_val}")

# ── 4. Confidence sweep ───────────────────────────────────────────────────────
print("\n  Running confidence threshold sweep...")
confs = np.arange(0.05, 0.71, 0.05)
sweep_r, sweep_mae, sweep_bias = [], [], []
for c in confs:
    pr = []
    for p in img_files:
        res = model(p, conf=float(c), iou=IOU, imgsz=IMGSZ, verbose=False)
        pr.append(len(res[0].boxes) if res[0].boxes is not None else 0)
    pr = np.array(pr, dtype=float)
    rr, _ = stats.pearsonr(gt, pr)
    sweep_r.append(rr)
    sweep_mae.append(np.mean(np.abs(pr - gt)))
    sweep_bias.append(np.mean(pr - gt))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, vals, title, ylabel in zip(
    axes, [sweep_r, sweep_mae, sweep_bias],
    ["Pearson r", "MAE", "Mean Bias"],
    ["r", "MAE (neurons)", "Bias (auto − manual)"]
):
    ax.plot(confs, vals, "o-", color="#2196F3", lw=2)
    ax.axvline(CONF, color="red", lw=1.2, linestyle="--", label=f"conf={CONF}")
    ax.set_xlabel("Confidence threshold", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Confidence Sweep — v7 model, whole-image inference",
             fontsize=12, fontweight="bold")
plt.tight_layout()
out_sweep = PAPER / "conf_sweep.png"
plt.savefig(str(out_sweep), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {out_sweep}")

# ── 5. Summary ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  MANUSCRIPT NUMBERS")
print("="*60)
print(f"  Model:    YOLO26m (v7) — trained on 424 images (2048x1708 px source)")
print(f"  Test set: {len(img_files)} images, {int(gt.sum())} neuron instances")
print(f"  mAP@50:   {map50:.3f}")
print(f"  Precision:{mp:.3f}  Recall:{mr:.3f}  F1:{f1:.3f}")
print(f"  Counting: r={r_:.3f}, MAE={mae:.2f}, bias={bias:+.2f}, SD={sd:.3f}, "
      f"95%LoA=[{loa_lo:.2f},{loa_hi:.2f}]")
