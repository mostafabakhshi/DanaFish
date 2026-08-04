from ultralytics import YOLO
import cv2
import numpy as np
import logging
import os

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
        self.confidence = 0.1  # Lowered to detect body in curved/difficult images
        self.overlap = 0.5
        
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
    
    def train_rotation_model(self, data_yaml_path, epochs=50):
        """Train the rotation detection model."""
        logger.info("Starting rotation model training...")
        
        # Initialize with YOLOv11 nano for faster training
        self.model = YOLO('yolo11n.pt')
        
        # Training configuration optimized for rotation detection
        results = self.model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=640,
            batch=8,
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
            save_period=10,
            device='cpu',
            workers=8,
            project="runs/detect",
            name="rotation_model_v1",
            exist_ok=True,
            pretrained=True,
            amp=True,
            fraction=1.0,
            cache=True,
            label_smoothing=0.0,  # No label smoothing for precise landmark detection
            verbose=True,
            seed=42
        )
        
        # Update model path to the best trained model
        self.model_path = f"runs/detect/rotation_model_v1/weights/best.pt"
        logger.info(f"Rotation model training completed. Best model saved to: {self.model_path}")
        
        return results
    
    def get_orientation_landmarks(self, image_path):
        """
        Detect head, tail, and body landmarks for orientation analysis.
        
        Returns:
            dict: Dictionary containing detected landmarks with their coordinates
        """
        if not self.model:
            raise ValueError("No rotation model loaded. Train or load a model first.")
        
        # Get predictions
        with open(os.devnull, 'w') as f:
            results = self.model.predict(image_path, conf=self.confidence, iou=self.overlap, verbose=False)
        
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
                
                # Assign to appropriate landmark category
                if class_name == 'h':  # head
                    if landmarks['head'] is None or confidence > landmarks['head']['confidence']:
                        landmarks['head'] = landmark_info
                elif class_name == 't':  # tail
                    if landmarks['tail'] is None or confidence > landmarks['tail']['confidence']:
                        landmarks['tail'] = landmark_info
                elif class_name == 'b':  # body
                    if landmarks['body'] is None or confidence > landmarks['body']['confidence']:
                        landmarks['body'] = landmark_info
                elif class_name == 'n':  # neurons
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
    # Test script for rotation model
    rotation_model = ZebraFishRotationModel()
    
    # Train the rotation model
    data_yaml = "datasets/fish13.v4-bnhtsize.yolov11/data.yaml"
    rotation_model.train_rotation_model(data_yaml, epochs=50)