"""Evaluate the neuron detector against manual annotations.

Loads the neuron detector named in config.py and reports, for one split of a
YOLO-format dataset:

  - detection metrics from the Ultralytics validation routine (mAP@50, mAP@50-95,
    precision, recall, F1), which sweeps the confidence threshold;
  - count agreement at the inference settings in config.py (confidence 0.35, NMS IoU
    0.5, imgsz 1280): mean absolute error, mean bias, 95% limits of agreement,
    Pearson correlation, and Lin's concordance correlation coefficient.

Each image is counted once, as the number of detections the detector returns for the
whole image, and compared with the number of annotated boxes in its label file. This
is the procedure behind the reported test-set evaluation.

Usage
    python evaluate.py --data path/to/data.yaml [--split test] [--out result.json]
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parent
IMAGE_EXT = ('.jpg', '.jpeg', '.png')
IMGSZ = 1280   # inference size used by model.py


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def split_image_dirs(data_yaml, split):
    """Image directories of one split, resolved the way data.yaml files are written."""
    data_yaml = Path(data_yaml).resolve()
    cfg = yaml.safe_load(data_yaml.read_text(encoding='utf-8'))
    if split not in cfg:
        raise SystemExit(f'{data_yaml} has no "{split}" split')
    base = Path(cfg['path']) if cfg.get('path') else data_yaml.parent
    if not base.is_absolute():
        base = (data_yaml.parent / base).resolve()
    entries = cfg[split] if isinstance(cfg[split], list) else [cfg[split]]
    dirs = [Path(e) if Path(e).is_absolute() else (base / e).resolve() for e in entries]
    for d in dirs:
        if not d.is_dir():
            raise SystemExit(f'image directory not found: {d}')
    return dirs


def labels_dir(images_dir):
    """YOLO convention: the label directory mirrors the image directory."""
    parts = list(images_dir.parts)
    if 'images' not in parts:
        raise SystemExit(f'cannot locate labels for {images_dir}: no "images" component')
    i = len(parts) - 1 - parts[::-1].index('images')
    parts[i] = 'labels'
    return Path(*parts)


def agreement(pred, ref):
    keys = sorted(set(pred) & set(ref))
    a = np.array([pred[k] for k in keys], float)
    m = np.array([ref[k] for k in keys], float)
    d = a - m
    sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
    ok = len(keys) > 2 and m.std() > 0 and a.std() > 0
    ccc = ((2 * np.cov(m, a)[0, 1]) / (m.var() + a.var() + (m.mean() - a.mean()) ** 2)
           if ok else float('nan'))
    return {'n_images': len(keys),
            'MAE': float(np.abs(d).mean()),
            'bias': float(d.mean()),
            'sd_of_differences': sd,
            'limits_of_agreement': [float(d.mean() - 1.96 * sd), float(d.mean() + 1.96 * sd)],
            'pearson_r': float(stats.pearsonr(m, a)[0]) if ok else float('nan'),
            'CCC': float(ccc),
            'total_automated': int(a.sum()),
            'total_manual': int(m.sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--data', required=True, help='YOLO data.yaml')
    ap.add_argument('--split', default='test', help='split to evaluate (default: test)')
    ap.add_argument('--out', help='optional path for a JSON copy of the results')
    args = ap.parse_args()

    os.chdir(ROOT)                       # config.py resolves weights relative to here
    sys.path.insert(0, str(ROOT))
    import config
    from ultralytics import YOLO

    weights = ROOT / config.NEURON_MODEL_PATH
    conf = config.MODEL_CONFIG['confidence']
    iou = config.MODEL_CONFIG['overlap']
    model = YOLO(str(weights))

    v = model.val(data=str(Path(args.data).resolve()), split=args.split, imgsz=IMGSZ,
                  verbose=False, plots=False)
    p, r = float(v.box.mp), float(v.box.mr)
    detection = {'mAP50': float(v.box.map50), 'mAP50_95': float(v.box.map),
                 'precision': p, 'recall': r, 'F1': 2 * p * r / (p + r) if (p + r) else 0.0}

    pred, ref = {}, {}
    for images in split_image_dirs(args.data, args.split):
        labels = labels_dir(images)
        for f in sorted(images.iterdir()):
            if f.suffix.lower() not in IMAGE_EXT:
                continue
            res = model.predict(str(f), imgsz=IMGSZ, conf=conf, iou=iou, verbose=False)[0]
            pred[f.stem] = int(len(res.boxes))
            lbl = labels / (f.stem + '.txt')
            if lbl.exists():
                ref[f.stem] = sum(1 for line in lbl.read_text().splitlines() if line.strip())

    result = {'weights': str(Path(config.NEURON_MODEL_PATH)), 'weights_md5': md5(weights),
              'data': str(Path(args.data).resolve()), 'split': args.split,
              'settings': {'confidence': conf, 'nms_iou': iou, 'imgsz': IMGSZ},
              'detection': detection, 'count_agreement': agreement(pred, ref)}

    a = result['count_agreement']
    print(f'weights   {result["weights"]}  (md5 {result["weights_md5"][:8]})')
    print(f'split     {args.split}: {a["n_images"]} annotated images')
    print(f'detection mAP@50 {detection["mAP50"]:.3f}  mAP@50-95 {detection["mAP50_95"]:.3f}  '
          f'precision {detection["precision"]:.3f}  recall {detection["recall"]:.3f}  '
          f'F1 {detection["F1"]:.3f}')
    print(f'counts    r {a["pearson_r"]:.3f}  MAE {a["MAE"]:.2f}  bias {a["bias"]:+.2f}  '
          f'LoA {a["limits_of_agreement"][0]:.2f} to {a["limits_of_agreement"][1]:+.2f}  '
          f'CCC {a["CCC"]:.3f}  totals {a["total_automated"]} vs {a["total_manual"]}')
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1), encoding='utf-8')
        print(f'written   {args.out}')


if __name__ == '__main__':
    main()
