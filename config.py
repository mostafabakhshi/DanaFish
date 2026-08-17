import os

# Project paths
HOME = os.getcwd()
DATASET_ZIP = "Zebra.v1i.coco.zip"
COCO_EXTRACT_PATH = "datasets/zebra_coco"
YOLO_OUTPUT_DIR = "datasets/zebra_yolo"
OUTPUT_ANNOTATED_PATH = "datasets/zebra_yolo/test/My_Prediction"
OUTPUT_EXCEL_PATH = "results/results_{}.xlsx"
NEURON_MODEL_PATH  = "runs/detect/runs/detect/neuron_v7_yolo26m_1280/weights/best.pt"  # YOLO26m neuron-only v7
TRAINED_MODEL_PATH = NEURON_MODEL_PATH

# ── Landmark detector (4-class: b, h, n, t) ──────────────────────────────────
# YOLO26m, trained at imgsz 1280 on datasets/fish13_union_4class — the pooled
# fish13.v4 and fish13.v8 annotation sets (700/99/65 images). Used for
# orientation standardization and to seed the spinal-cord region.
LANDMARK_MODEL_PATH = "runs/detect/runs/detect/landmark_v8_yolo26m_union/weights/best.pt"

# Preprocessing configuration (pipeline Step 1, before any inference)
PREPROCESS_CONFIG = {
    "enabled": True,       # Subtract the per-image background pedestal.
    "percentile": 1.0,     # Percentile of non-zero pixels taken as the black point.
    "channel": 1,          # Green channel — carries the GFP signal.
    "min_pedestal": 15.0,  # Below this the image has no pedestal worth removing and
                           # is passed through untouched. Guards against acting on
                           # images that do not exhibit the failure mode.
}

# Landmark detector configuration
LANDMARK_CONFIG = {
    "confidence": 0.1,        # Head/tail/neuron classes.
    "body_confidence": 0.01,  # The body class scores lower on coiled and dim larvae,
                              # so it carries its own, lower threshold. Head and tail
                              # keep 0.1, so orientation is unaffected.
    "overlap": 0.5,           # NMS IoU.
    "imgsz": 960,             # Default single-scale inference size.
    # The body/spine class is scale-sensitive on elongated and coiled larvae, so
    # the body is searched over several inference sizes in this order, taking the
    # first detection at or above body_search_accept and otherwise the most
    # confident one found. Head and tail are read from the same pass, so the
    # accepted scale determines all landmarks for that image.
    "body_search_scales": (960, 416, 320),
    "body_search_accept": 0.1,
}

# Model configuration
MODEL_CONFIG = {
    "confidence": 0.35,          # Selected on the validation split: within the flat-MAE region
                                 # (0.30-0.38) it is the threshold with bias closest to zero.
    "overlap": 0.5,             # Adjusted for better balance
    "threshold": 30,
    "padding": 20,
    "vertical_consistency": 10,
    "dbscan_eps": 11.99,           # Reduced from 21 to 15 for more precise horizontal clustering
    "dbscan_min_samples": 1,
    # Additional model parameters
    "max_det": 300,             # Maximum number of detections
    "agnostic_nms": True,       # Class-agnostic NMS
    "multi_scale": True,        # Multi-scale training
    "rect": False,              # Rectangular training
    "dropout": 0.2,             # Increased dropout
    "fraction": 1.0,            # Use full dataset
    "cache": True,              # Cache images
    "device": 0,                # Use GPU
    "workers": 8,               # Number of worker threads
    "project": "runs/detect",   # Project directory
    "name": "train_enhanced_v2", # Run name
    "exist_ok": True,           # Overwrite existing experiment
    "pretrained": True,         # Use pretrained weights
    "amp": True,                # Use automatic mixed precision
    "label_smoothing": 0.1,     # Label smoothing
    "nbs": 64,                  # Nominal batch size
    "overlap_mask": True,       # Enable mask overlap
    "mask_ratio": 4,            # Mask ratio
    "single_cls": False,        # Multi-class training
    "verbose": True,            # Enable verbose output
    "seed": 42                  # Random seed
}

# Class colors for visualization (BGR format)
CLASS_COLORS = {
    1: (0, 165, 255),    # Orange for "n" (more visible than pure red)
    0: (255, 191, 0)     # Deep sky blue for "b" (more visible than pure blue)
}

# Data configuration
DATA_CONFIG = {
    "nc": 2,
    "names": ['b', 'n']
}

# Create necessary directories
os.makedirs(os.path.join(YOLO_OUTPUT_DIR, "train", "labels"), exist_ok=True)
os.makedirs(os.path.join(YOLO_OUTPUT_DIR, "valid", "labels"), exist_ok=True)
os.makedirs(os.path.join(YOLO_OUTPUT_DIR, "test", "labels"), exist_ok=True)
os.makedirs(OUTPUT_ANNOTATED_PATH, exist_ok=True) 