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

    def train(self, data_yaml_path):
        """Train the model with the given data configuration."""
        logger.info("Starting model training...")
        self.model.train(
            data=data_yaml_path,
            epochs=50,
            imgsz=640,
            batch=8,
            optimizer="AdamW",
            lr0=0.0005,
            lrf=0.005,
            momentum=0.937,
            weight_decay=0.001,
            warmup_epochs=5.0,
            warmup_momentum=0.8,
            box=5.0,
            cls=0.3,
            dfl=1.0,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=15.0,
            translate=0.2,
            scale=0.5,
            shear=2.0,
            perspective=0.0,
            flipud=0.1,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.3,
            copy_paste=0.3,
            auto_augment="randaugment",
            erasing=0.4,
            close_mosaic=15,
            cos_lr=True,
            patience=100,
            save_period=5,
            device=0,
            workers=8,
            project="runs/detect",
            name="train_enhanced_v2",
            exist_ok=True,
            pretrained=True,
            amp=True,
            multi_scale=True,
            rect=False,
            overlap_mask=True,
            mask_ratio=4,
            dropout=0.2,
            fraction=1.0,
            cache=True,
            label_smoothing=0.1,
            nbs=64,
            single_cls=False,
            verbose=True,
            seed=42
        )
        logger.info("Training completed")

    def validate(self, data_yaml_path):
        """Validate the model on the given dataset."""
        logger.info("Starting model validation...")
        results = self.model.val(data=data_yaml_path)
        logger.info("Validation completed")
        return results 