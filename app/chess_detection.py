import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time

class ChessDetectionService:
    def __init__(self, model_path='app/model/best.pt'):
        try:
            self.model = YOLO(model_path)
            print(f"YOLO model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.model = None
        
        self.detection_active = False
        self.detection_thread = None
        self.camera_index = 0
        self.detection_mode = 'raw'
        self.show_bbox = True
        self.cap = None
        self.fps_counter = 0
        self.last_fps_time = time.time()
        
    def apply_clahe(self, image):
        """Apply CLAHE enhancement to image"""
        try:
            if len(image.shape) == 3:
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l_channel, a_channel, b_channel = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                l_channel_clahe = clahe.apply(l_channel)
                lab_clahe = cv2.merge([l_channel_clahe, a_channel, b_channel])
                enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
                return enhanced
            else:
                return image
        except Exception as e:
            print(f"CLAHE error: {e}")
            return image
    
    def crop_to_square(self, image, size=720):
        """Crop image to square and resize"""
        try:
            if image is None:
                return None
                
            h, w = image.shape[:2]
            
            if h > w:
                start = (h - w) // 2
                cropped = image[start:start + w, :].copy()
            elif w > h:
                start = (w - h) // 2
                cropped = image[:, start:start + h].copy()
            else:
                cropped = image.copy()
            
            resized = cv2.resize(cropped, (size, size))
            return resized
            
        except Exception as e:
            print(f"Crop error: {e}")
            return image
    
    def detect_pieces_realtime(self, image):
        """Detect pieces and return processed image"""
        if self.model is None or image is None:
            return image
            
        try:
            input_image = image.copy()
            processed_image = self.crop_to_square(input_image, 720)
            
            if processed_image is None:
                return image
            
            if self.detection_mode == 'clahe':
                processed_image = self.apply_clahe(processed_image)
            
            # Skip detection for some frames to improve performance
            if self.fps_counter % 3 == 0:  # Only detect every 3rd frame
                results = self.model(processed_image, verbose=False)
                
                if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                    if self.show_bbox:
                        self.last_detection_result = results[0].plot()
                        return self.last_detection_result
                    else:
                        self.last_detection_result = processed_image
                        return processed_image
                else:
                    self.last_detection_result = processed_image
                    return processed_image
            else:
                # Return last detection result for performance
                return getattr(self, 'last_detection_result', processed_image)
                
        except Exception as e:
            print(f"Detection error: {e}")
            return image
    
    def start_opencv_detection(self, camera_index=0, mode='raw', show_bbox=True):
        """Start real-time detection in OpenCV window"""
        self.camera_index = camera_index
        self.detection_mode = mode
        self.show_bbox = show_bbox
        
        if self.detection_active:
            self.stop_opencv_detection()
        
        self.detection_active = True
        self.detection_thread = threading.Thread(target=self._detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        return True
    
    def _detection_loop(self):
        """Main detection loop running in separate thread"""
        try:
            # Try different backends for better compatibility
            backends_to_try = [
                cv2.CAP_DSHOW,      # DirectShow (Windows)
                cv2.CAP_MSMF,       # Media Foundation (Windows)
                cv2.CAP_V4L2,       # Video4Linux2 (Linux)
                cv2.CAP_ANY         # Any available
            ]
            
            self.cap = None
            for backend in backends_to_try:
                try:
                    print(f"Trying camera backend: {backend}")
                    self.cap = cv2.VideoCapture(self.camera_index, backend)
                    if self.cap.isOpened():
                        print(f"Successfully opened camera with backend: {backend}")
                        break
                    else:
                        if self.cap:
                            self.cap.release()
                        self.cap = None
                except Exception as e:
                    print(f"Backend {backend} failed: {e}")
                    continue
            
            if not self.cap or not self.cap.isOpened():
                print(f"Error: Could not open camera {self.camera_index} with any backend")
                self.detection_active = False
                return
            
            # Configure camera with error handling
            try:
                # Set buffer size to 1 to reduce latency
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Set resolution
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                
                # Set FPS
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                
                # Disable auto-exposure for better performance (if supported)
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                
            except Exception as e:
                print(f"Warning: Could not set camera properties: {e}")
            
            # Get actual camera properties
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            print(f"Camera {self.camera_index} opened successfully")
            print(f"Resolution: {actual_width}x{actual_height}")
            print(f"FPS: {actual_fps}")
            print(f"Detection mode: {self.detection_mode}")
            print(f"Show bounding boxes: {self.show_bbox}")
            print("Controls: 'q' to quit, 'space' to toggle bbox, 'm' to toggle mode")
            
            # Create window
            cv2.namedWindow('Chess Detection - ChessMon', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Chess Detection - ChessMon', 720, 720)
            
            # FPS tracking
            frame_count = 0
            start_time = time.time()
            last_frame_time = time.time()
            
            # Main loop
            while self.detection_active:
                current_time = time.time()
                
                # Add frame rate limiting to prevent overwhelming
                if current_time - last_frame_time < 0.033:  # ~30 FPS max
                    time.sleep(0.001)
                    continue
                
                last_frame_time = current_time
                
                # Try to read frame with timeout
                ret = False
                frame = None
                
                try:
                    ret, frame = self.cap.read()
                except Exception as e:
                    print(f"Frame read error: {e}")
                    ret = False
                
                if not ret or frame is None:
                    print("Warning: Could not read frame, retrying...")
                    time.sleep(0.1)
                    
                    # Try to reset camera if too many failures
                    frame_count += 1
                    if frame_count % 10 == 0:
                        print("Attempting to reset camera...")
                        try:
                            self.cap.release()
                            time.sleep(0.5)
                            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                            if not self.cap.isOpened():
                                print("Camera reset failed")
                                break
                        except Exception as e:
                            print(f"Camera reset error: {e}")
                            break
                    continue
                
                # Reset frame count on successful read
                frame_count = 0
                self.fps_counter += 1
                
                # Process frame for detection
                try:
                    processed_frame = self.detect_pieces_realtime(frame)
                    display_frame = self._add_info_overlay(processed_frame)
                except Exception as e:
                    print(f"Frame processing error: {e}")
                    display_frame = frame
                
                # Display frame
                try:
                    cv2.imshow('Chess Detection - ChessMon', display_frame)
                except Exception as e:
                    print(f"Display error: {e}")
                    break
                
                # Handle key presses
                try:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Quit requested by user")
                        break
                    elif key == ord(' '):
                        self.show_bbox = not self.show_bbox
                        print(f"Bounding boxes: {'ON' if self.show_bbox else 'OFF'}")
                    elif key == ord('m'):
                        self.detection_mode = 'clahe' if self.detection_mode == 'raw' else 'raw'
                        print(f"Detection mode: {self.detection_mode}")
                    elif key == ord('r'):  # Reset camera
                        print("Manual camera reset...")
                        self.cap.release()
                        time.sleep(0.5)
                        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                except Exception as e:
                    print(f"Key handling error: {e}")
                
                # Calculate and display FPS every 30 frames
                if self.fps_counter % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = 30 / elapsed if elapsed > 0 else 0
                    print(f"FPS: {fps:.1f}")
                    start_time = time.time()
            
        except Exception as e:
            print(f"Detection loop error: {e}")
        finally:
            # Cleanup
            try:
                if self.cap and self.cap.isOpened():
                    self.cap.release()
                cv2.destroyAllWindows()
            except Exception as e:
                print(f"Cleanup error: {e}")
            
            self.detection_active = False
            print("Detection stopped")
    
    def _add_info_overlay(self, frame):
        """Add information overlay to frame"""
        try:
            if frame is None:
                return frame
                
            display_frame = frame.copy()
            h, w = display_frame.shape[:2]
            
            # Calculate FPS
            current_time = time.time()
            if hasattr(self, 'last_fps_time'):
                fps = 1.0 / (current_time - self.last_fps_time) if (current_time - self.last_fps_time) > 0 else 0
            else:
                fps = 0
            self.last_fps_time = current_time
            
            # Background for text
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (10, 10), (350, 140), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
            
            # Add text information
            cv2.putText(display_frame, f"Camera: {self.camera_index}", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Mode: {self.detection_mode.upper()}", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_frame, f"BBox: {'ON' if self.show_bbox else 'OFF'}", (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (20, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(display_frame, "Q:quit | Space:bbox | M:mode | R:reset", (20, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.putText(display_frame, f"Frame: {self.fps_counter}", (20, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            return display_frame
            
        except Exception as e:
            print(f"Overlay error: {e}")
            return frame
    
    def stop_opencv_detection(self):
        """Stop real-time detection"""
        print("Stopping detection...")
        self.detection_active = False
        
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=5)
        
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Cleanup error: {e}")
        
        print("Detection stopped by user")
    
    def update_detection_settings(self, camera_index=None, mode=None, show_bbox=None):
        """Update detection settings during runtime"""
        if camera_index is not None:
            self.camera_index = camera_index
        if mode is not None:
            self.detection_mode = mode
        if show_bbox is not None:
            self.show_bbox = show_bbox
        
        print(f"Settings updated - Camera: {self.camera_index}, Mode: {self.detection_mode}, BBox: {self.show_bbox}")
    
    def is_detection_active(self):
        """Check if detection is currently running"""
        return self.detection_active
    
    # Keep existing methods for web-based detection
    def detect_pieces(self, image, mode='raw', show_bbox=True):
        """Detect pieces for web API"""
        if self.model is None:
            print("YOLO model not loaded")
            return image, None
            
        try:
            input_image = image.copy()
            processed_image = self.crop_to_square(input_image, 720)
            
            if processed_image is None:
                return image, None
            
            if mode == 'clahe':
                processed_image = self.apply_clahe(processed_image)
            
            results = self.model(processed_image, verbose=False)
            
            if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                if show_bbox:
                    annotated_frame = results[0].plot()
                    return annotated_frame, results[0]
                else:
                    return processed_image, results[0]
            else:
                return processed_image, None
                
        except Exception as e:
            print(f"Web detection error: {e}")
            return image, None
    
    def get_detection_data(self, results):
        """Extract detection data from YOLO results"""
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