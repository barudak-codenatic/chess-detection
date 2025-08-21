import cv2
import numpy as np
from ultralytics import YOLO
import base64
from io import BytesIO
from PIL import Image

class ChessDetectionService:
    def __init__(self, model_path='app/model/best.pt'):
        try:
            self.model = YOLO(model_path)
            print(f"YOLO model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.model = None
        self.detection_active = False
        
    def apply_clahe(self, image):
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab_planes = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        lab_planes[0] = clahe.apply(lab_planes[0])
        
        # Merge and convert back to RGB
        lab = cv2.merge(lab_planes)
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return enhanced
    
    def crop_to_square(self, image, size=720):
        h, w = image.shape[:2]
        # Center crop to square
        if h > w:
            start = (h - w) // 2
            image = image[start:start + w, :]
        elif w > h:
            start = (w - h) // 2
            image = image[:, start:start + h]
        
        # Resize to target size
        image = cv2.resize(image, (size, size))
        return image
    
    def detect_pieces(self, image, mode='raw', show_bbox=True):
        if self.model is None:
            print("YOLO model not loaded")
            return image, None
            
        try:
            # Crop to 720x720 square
            processed_image = self.crop_to_square(image, 720)
            
            # Apply preprocessing based on mode
            if mode == 'clahe':
                processed_image = self.apply_clahe(processed_image)
            
            # Run YOLO detection
            results = self.model(processed_image, verbose=False)
            
            # Handle results
            if len(results) > 0 and results[0].boxes is not None:
                if show_bbox:
                    # Draw bounding boxes
                    annotated_frame = results[0].plot()
                    return annotated_frame, results[0]
                else:
                    # Return processed image without boxes
                    return processed_image, results[0]
            else:
                # No detections found
                return processed_image, None
                
        except Exception as e:
            print(f"Detection error: {e}")
            return image, None
    
    def get_detection_data(self, results):
        if results is None or results.boxes is None:
            return []
        
        detections = []
        try:
            for box in results.boxes:
                detection = {
                    'class': int(box.cls[0]),
                    'confidence': float(box.conf[0]),
                    'bbox': box.xyxy[0].tolist(),
                    'class_name': self.model.names[int(box.cls[0])] if hasattr(self.model, 'names') else f"Class_{int(box.cls[0])}"
                }
                detections.append(detection)
        except Exception as e:
            print(f"Error processing detections: {e}")
        
        return detections