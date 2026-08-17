from ultralytics import YOLO
import cv2
import numpy as np
import logging
import os

from config import LANDMARK_CONFIG, LANDMARK_MODEL_PATH

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ZebraFishRotationModel:
    """
    Model for detecting zebrafish orientation landmarks (head, tail, body)
    to correct image rotation and ensure standard orientation.
    """
    
    def __init__(self, model_path=None):
        """Initialize the rotation model."""
        self.model_path = model_path
        self.model = None
        # Thresholds and inference size come from config so there is one source of
        # truth. imgsz must match the resolution the weights were trained at.
        self.confidence = LANDMARK_CONFIG["confidence"]
        # Separate, lower threshold for the BODY class only. The body/spinal-cord
        # class scores lower on coiled and dim larvae. Head/tail keep the higher
        # threshold, so orientation is unaffected by this.
        self.body_confidence = LANDMARK_CONFIG["body_confidence"]
        self.overlap = LANDMARK_CONFIG["overlap"]
        self.imgsz = LANDMARK_CONFIG["imgsz"]
        self.body_search_scales = tuple(LANDMARK_CONFIG.get("body_search_scales")
                                        or (LANDMARK_CONFIG["imgsz"],))
        self.body_search_accept = LANDMARK_CONFIG.get("body_search_accept", 0.1)

        # Class mapping for the new dataset
        self.class_names = {
            0: 'b',  # body (spine)
            1: 'h',  # head
            2: 'n',  # neurons (ignore for rotation)
            3: 't'   # tail
        }
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            logger.info("No trained rotation model found. Use train_rotation_model() first.")
    
    def load_model(self, model_path):
        """Load a trained rotation model."""
        logger.info(f"Loading rotation model from: {model_path}")
        self.model = YOLO(model_path)
        self.model_path = model_path
        logger.info("Rotation model loaded successfully")
    
    def train_rotation_model(self, data_yaml_path, epochs=300, base="yolo26m.pt",
                             imgsz=1280, batch=2,
                             run_name="landmark_v8_yolo26m_union"):
        """Train the landmark detection model.

        The defaults reproduce the shipped weights: YOLO26m at imgsz 1280 on the
        pooled 4-class dataset (datasets/fish13_union_4class/data.yaml). The full
        argument set of the shipped run is also preserved in that run's own
        args.yaml.
        """
        logger.info(f"Starting landmark model training from {base} at imgsz {imgsz}...")

        self.model = YOLO(base)

        # Training configuration optimized for landmark detection
        results = self.model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,      # No rotation augmentation since we want to learn orientation
            translate=0.1,
            scale=0.5,
            shear=0.0,        # No shear for rotation detection
            perspective=0.0,
            flipud=0.0,       # No vertical flip for orientation learning
            fliplr=0.0,       # No horizontal flip for orientation learning
            mosaic=1.0,
            mixup=0.1,
            copy_paste=0.1,
            auto_augment="randaugment",
            erasing=0.2,
            close_mosaic=10,
            cos_lr=True,
            patience=50,
            save_period=25,
            device=0,
            # Ultralytics resolves a relative project against its own runs_dir, which
            # would nest the run a second level deep; give it an absolute path.
            workers=4,
            project=os.path.abspath(os.path.join("runs", "detect", "runs", "detect")),
            name=run_name,
            exist_ok=True,
            pretrained=True,
            amp=True,
            fraction=1.0,
            cache=False,
            label_smoothing=0.0,  # No label smoothing for precise landmark detection
            verbose=True,
            seed=42
        )
        
        # Update model path to the best trained model
        self.model_path = f"runs/detect/runs/detect/{run_name}/weights/best.pt"
        logger.info(f"Landmark model training completed. Best model saved to: {self.model_path}")
        
        return results
    
    def get_orientation_landmarks(self, image_path):
        """
        Detect head, tail, and body landmarks for orientation analysis.

        The body/spine class is scale-sensitive on elongated and coiled larvae, so
        it is searched over the inference sizes in LANDMARK_CONFIG
        ['body_search_scales']. The first scale returning a body at or above
        ['body_search_accept'] is taken; if none reaches it, the most confident
        body found across the scales is used. Head and tail come from the same
        pass as the accepted body, so all landmarks stay mutually consistent.

        Returns:
            dict: Dictionary containing detected landmarks with their coordinates
        """
        if not self.model:
            raise ValueError("No rotation model loaded. Train or load a model first.")

        scales = self.body_search_scales or (self.imgsz,)
        best = None
        last = None
        for size in scales:
            last = self._detect_at(image_path, size)
            body = last.get('body')
            if body:
                if best is None or body['confidence'] > best['body']['confidence']:
                    best = last
                if body['confidence'] >= self.body_search_accept:
                    break
        return best if best is not None else last

    def _detect_at(self, image_path, imgsz):
        """Run one detection pass at a given inference size."""
        # Predict down to the lowest of the per-class thresholds; each class is
        # then filtered at its own threshold below.
        floor_conf = min(self.confidence, self.body_confidence)
        with open(os.devnull, 'w') as f:
            results = self.model.predict(image_path, conf=floor_conf, iou=self.overlap,
                                         imgsz=imgsz, verbose=False)

        landmarks = {
            'head': None,
            'tail': None,
            'body': None,
            'neurons': []  # Store but ignore for rotation
        }
        
        image = cv2.imread(image_path)
        image_height, image_width = image.shape[:2]
        
        for result in results:
            for item in result.boxes:
                # Get bounding box coordinates
                x, y, w, h = item.xywh[0].tolist()
                start_x = max(0, min(int(x - w / 2), image_width - 1))
                start_y = max(0, min(int(y - h / 2), image_height - 1))
                end_x = max(0, min(int(x + w / 2), image_width - 1))
                end_y = max(0, min(int(y + h / 2), image_height - 1))
                
                # Get class and confidence
                class_id = int(item.cls[0])
                confidence = float(item.conf[0])
                class_name = self.class_names[class_id]
                
                # Store landmark information
                landmark_info = {
                    'bbox': [start_x, start_y, end_x, end_y],
                    'center': [int(x), int(y)],
                    'confidence': confidence
                }
                
                # Assign to appropriate landmark category, applying the
                # per-class threshold: body uses self.body_confidence, every
                # other class keeps self.confidence.
                if class_name == 'h':  # head
                    if confidence < self.confidence:
                        continue
                    if landmarks['head'] is None or confidence > landmarks['head']['confidence']:
                        landmarks['head'] = landmark_info
                elif class_name == 't':  # tail
                    if confidence < self.confidence:
                        continue
                    if landmarks['tail'] is None or confidence > landmarks['tail']['confidence']:
                        landmarks['tail'] = landmark_info
                elif class_name == 'b':  # body
                    if confidence < self.body_confidence:
                        continue
                    if landmarks['body'] is None or confidence > landmarks['body']['confidence']:
                        landmarks['body'] = landmark_info
                elif class_name == 'n':  # neurons
                    if confidence < self.confidence:
                        continue
                    landmarks['neurons'].append(landmark_info)
        
        return landmarks
    
    def validate_rotation_model(self, data_yaml_path):
        """Validate the rotation detection model."""
        if not self.model:
            raise ValueError("No rotation model loaded.")
        
        logger.info("Starting rotation model validation...")
        results = self.model.val(data=data_yaml_path)
        logger.info("Rotation model validation completed")
        return results

if __name__ == "__main__":
    # Retrain the landmark detector from scratch.
    # On Windows the dataloader workers re-import this module, so the training call
    # must stay behind this guard.
    rotation_model = ZebraFishRotationModel()
    data_yaml = "datasets/fish13_union_4class/data.yaml"
    rotation_model.train_rotation_model(data_yaml, epochs=300)