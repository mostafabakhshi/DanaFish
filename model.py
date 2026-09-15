from ultralytics import YOLO
import cv2
import numpy as np
from config import MODEL_CONFIG, NEURON_MODEL_PATH
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ZebraFishModel:
    def __init__(self):
        logger.info("Loading neuron model v7...")
        self.model      = YOLO(NEURON_MODEL_PATH)
        self.confidence = MODEL_CONFIG["confidence"]
        self.overlap    = MODEL_CONFIG["overlap"]
        logger.info("Model loaded successfully")

    def get_predictions(self, image_path):
        """Detect neurons with v7 model. Returns label=1 for all detections (neuron class)."""
        image = cv2.imread(image_path)
        image_height, image_width = image.shape[:2]

        results = self.model.predict(
            image_path, conf=self.confidence, iou=self.overlap, imgsz=1280, verbose=False)

        boxes, labels, confidences = [], [], []
        for result in results:
            if result.boxes is None:
                continue
            for item in result.boxes:
                x, y, w, h = item.xywh[0].tolist()
                x1 = max(0, min(int(x - w / 2), image_width - 1))
                y1 = max(0, min(int(y - h / 2), image_height - 1))
                x2 = max(0, min(int(x + w / 2), image_width - 1))
                y2 = max(0, min(int(y + h / 2), image_height - 1))
                boxes.append([x1, y1, x2, y2])
                labels.append(1)   # neuron → class 1 for ExactBodyRegionAnalyzer
                confidences.append(item.conf[0])

        return labels, np.array(boxes) if boxes else np.array([]).reshape(0, 4), confidences

    def train(self, data_yaml_path, base="yolo26m.pt", run_name="neuron_retrain"):
        """Train a neuron detector with the settings of the released model.

        Starts from the COCO-pretrained YOLO26m checkpoint, not from the released
        weights, and uses the hyperparameters the released neuron detector was
        trained with. A new run directory is written under runs/detect; an existing
        run of the same name is not overwritten.
        """
        logger.info(f"Starting neuron detector training from {base}...")
        return YOLO(base).train(
            data=data_yaml_path,
            project=os.path.abspath(os.path.join("runs", "detect")),
            name=run_name,
            exist_ok=False,
            pretrained=True,
            imgsz=1280,
            batch=2,
            nbs=64,
            optimizer="AdamW",
            lr0=0.0003,
            lrf=0.01,
            weight_decay=0.0005,
            epochs=150,
            patience=50,
            seed=42,
            deterministic=True,
            amp=True,
            cos_lr=False,
            close_mosaic=20,
            degrees=10.0,
            translate=0.1,
            scale=0.4,
            flipud=0.3,
            fliplr=0.5,
            mosaic=0.5,
            mixup=0.0,
            copy_paste=0.3,
            erasing=0.2,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            workers=4,
            device=0,
        )

    def validate(self, data_yaml_path):
        """Validate the model on the given dataset."""
        logger.info("Starting model validation...")
        results = self.model.val(data=data_yaml_path)
        logger.info("Validation completed")
        return results 