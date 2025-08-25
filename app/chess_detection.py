import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time
from scipy import ndimage
from sklearn.cluster import DBSCAN

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
        
        # Board detection attributes
        self.board_corners = None
        self.grid_points = None
        self.board_detection_enabled = True
        self.show_board_grid = True
        
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

    # ========== BOARD DETECTION METHODS FROM CHESSBOARD-DETECTION.IPYNB ==========
    
    def detect_board_canny_minimal(self, image):
        """Preprocessing minimal untuk fokus ke deteksi garis"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            edges = cv2.Canny(blurred, 30, 90, apertureSize=3)
            return edges
        except Exception as e:
            print(f"Edge detection error: {e}")
            return None

    def detect_lines_hough(self, edges):
        """Detect horizontal and vertical lines using Hough transform"""
        try:
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=150)
            horizontal_lines = []
            vertical_lines = []

            if lines is not None:
                for line in lines:
                    rho, theta = line[0]
                    # Perketat toleransi untuk hanya horizontal dan vertikal murni
                    if abs(theta) < np.pi/12 or abs(theta - np.pi) < np.pi/12:
                        horizontal_lines.append((rho, theta))
                    elif abs(theta - np.pi/2) < np.pi/12:
                        vertical_lines.append((rho, theta))

            return horizontal_lines, vertical_lines
        except Exception as e:
            print(f"Hough lines error: {e}")
            return [], []

    def cluster_lines(self, lines, tolerance=15):
        """Cluster similar lines together"""
        try:
            if not lines:
                return []

            rhos = np.array([line[0] for line in lines]).reshape(-1, 1)
            clustering = DBSCAN(eps=tolerance, min_samples=1).fit(rhos)

            clustered_lines = []
            for cluster_id in set(clustering.labels_):
                cluster_lines = [lines[i] for i in range(len(lines)) if clustering.labels_[i] == cluster_id]
                # Pilih garis pertama di cluster
                chosen_line = cluster_lines[0]
                clustered_lines.append(chosen_line)

            return sorted(clustered_lines, key=lambda x: x[0])
        except Exception as e:
            print(f"Clustering error: {e}")
            return lines

    def complete_lines_to_grid(self, lines, is_horizontal=True, image_shape=None):
        """Melengkapi garis menjadi 9 garis untuk membentuk 8 petak"""
        try:
            if len(lines) == 0:
                return []
            
            if len(lines) >= 9:
                # Jika sudah cukup, ambil 9 yang terdistribusi merata
                indices = np.linspace(0, len(lines)-1, 9, dtype=int)
                return [lines[i] for i in indices]
            
            # Jika kurang dari 9, hitung jarak rata-rata dan tambahkan garis
            sorted_lines = sorted(lines, key=lambda x: x[0])
            
            if len(sorted_lines) < 2:
                return sorted_lines
            
            # Hitung jarak rata-rata antar garis
            distances = []
            for i in range(1, len(sorted_lines)):
                distances.append(abs(sorted_lines[i][0] - sorted_lines[i-1][0]))
            
            if len(distances) == 0:
                return sorted_lines
                
            avg_distance = np.mean(distances)
            
            completed_lines = sorted_lines.copy()
            
            # Tambahkan garis di awal jika perlu
            while len(completed_lines) < 9:
                first_rho = completed_lines[0][0]
                new_rho = first_rho - avg_distance
                
                # Cek apakah masih dalam batas gambar
                if new_rho > 0:
                    theta = completed_lines[0][1]
                    completed_lines.insert(0, (new_rho, theta))
                else:
                    break
            
            # Tambahkan garis di akhir jika masih kurang
            while len(completed_lines) < 9:
                last_rho = completed_lines[-1][0]
                new_rho = last_rho + avg_distance
                
                # Cek batas gambar
                max_limit = image_shape[0] if is_horizontal else image_shape[1]
                if new_rho < max_limit:
                    theta = completed_lines[-1][1]
                    completed_lines.append((new_rho, theta))
                else:
                    break
            
            return completed_lines[:9]
        except Exception as e:
            print(f"Complete lines error: {e}")
            return lines

    def line_intersections(self, h_lines, v_lines):
        """Calculate intersections between horizontal and vertical lines"""
        try:
            intersections = []

            for h_rho, h_theta in h_lines:
                for v_rho, v_theta in v_lines:
                    A = np.array([[np.cos(h_theta), np.sin(h_theta)],
                                 [np.cos(v_theta), np.sin(v_theta)]])
                    b = np.array([h_rho, v_rho])

                    try:
                        point = np.linalg.solve(A, b)
                        intersections.append((int(point[0]), int(point[1])))
                    except np.linalg.LinAlgError:
                        continue

            return intersections
        except Exception as e:
            print(f"Line intersections error: {e}")
            return []

    def detect_board_corners(self, intersections, image_shape):
        """Detect 4 main corners of the chessboard"""
        try:
            if len(intersections) < 4:
                return None

            points = np.array(intersections)

            # Cari batas terluar
            min_x, min_y = np.min(points, axis=0)
            max_x, max_y = np.max(points, axis=0)

            # Fungsi cari titik terdekat
            def closest_point(target):
                distances = np.linalg.norm(points - target, axis=1)
                return tuple(points[np.argmin(distances)])

            # Cari masing-masing sudut
            top_left = closest_point((min_x, min_y))
            top_right = closest_point((max_x, min_y))
            bottom_right = closest_point((max_x, max_y))
            bottom_left = closest_point((min_x, max_y))

            corners = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
            return corners
        except Exception as e:
            print(f"Corner detection error: {e}")
            return None

    def apply_homography(self, image, corners):
        """Apply homography transformation to flatten the board"""
        try:
            if corners is None:
                return None

            # Sort corners: top-left, top-right, bottom-right, bottom-left
            rect = np.zeros((4, 2), dtype=np.float32)
            s = corners.sum(axis=1)
            rect[0] = corners[np.argmin(s)]  # top-left
            rect[2] = corners[np.argmax(s)]  # bottom-right

            diff = np.diff(corners, axis=1)
            rect[1] = corners[np.argmin(diff)]  # top-right
            rect[3] = corners[np.argmax(diff)]  # bottom-left

            # Define destination points for 720x720 square (match our crop size)
            dst = np.array([[0, 0], [720, 0], [720, 720], [0, 720]], dtype=np.float32)

            # Compute homography matrix
            M = cv2.getPerspectiveTransform(rect, dst)

            # Apply perspective transformation
            flattened = cv2.warpPerspective(image, M, (720, 720))

            return flattened
        except Exception as e:
            print(f"Homography error: {e}")
            return None

    def generate_grid_coordinates(self, size=720):
        """Generate chess square coordinates"""
        try:
            coords = {}
            files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
            ranks = ['8', '7', '6', '5', '4', '3', '2', '1']

            square_size = size // 8

            for i in range(8):
                for j in range(8):
                    file_name = files[j]
                    rank_name = ranks[i]
                    square_name = file_name + rank_name

                    x = j * square_size + square_size // 2
                    y = i * square_size + square_size // 2

                    coords[square_name] = (x, y)

            return coords
        except Exception as e:
            print(f"Grid coordinates error: {e}")
            return {}

    def detect_chessboard(self, image):
        """Complete chessboard detection pipeline"""
        try:
            if image is None:
                return None, None, None, None
            
            # Step 1: Edge Detection
            edges = self.detect_board_canny_minimal(image)
            if edges is None:
                return None, None, None, None

            # Step 2: Hough Line Transform
            h_lines_raw, v_lines_raw = self.detect_lines_hough(edges)

            # Step 3: Cluster lines
            h_lines_clustered = self.cluster_lines(h_lines_raw, tolerance=15)
            v_lines_clustered = self.cluster_lines(v_lines_raw, tolerance=15)

            # Step 4: Complete lines to 9 (untuk 8 petak)
            h_lines_complete = self.complete_lines_to_grid(h_lines_clustered, True, image.shape)
            v_lines_complete = self.complete_lines_to_grid(v_lines_clustered, False, image.shape)

            # Step 5: Line Intersections
            intersections = self.line_intersections(h_lines_complete, v_lines_complete)

            # Step 6: Detect board corners
            corners = self.detect_board_corners(intersections, image.shape)

            # Step 7: Apply homography to get flattened board
            flattened_board = self.apply_homography(image, corners)

            # Step 8: Generate grid coordinates (untuk flattened board)
            grid_coords = self.generate_grid_coordinates()

            return corners, intersections, grid_coords, flattened_board

        except Exception as e:
            print(f"Chessboard detection error: {e}")
            return None, None, None, None

    def draw_chessboard_overlay(self, image, corners, grid_coords, flattened_board=None, use_flattened=True):
        """Draw chessboard grid overlay on image"""
        try:
            if image is None:
                return image

            # Jika ada flattened board dan kita ingin menggunakan flattened view
            if use_flattened and flattened_board is not None and grid_coords is not None:
                overlay_image = flattened_board.copy()
                
                # Draw grid lines (8x8 grid on 720x720 image)
                square_size = 720 // 8
                for i in range(9):
                    # Vertical lines
                    cv2.line(overlay_image, (i * square_size, 0), (i * square_size, 720), (255, 0, 0), 2)
                    # Horizontal lines
                    cv2.line(overlay_image, (0, i * square_size), (720, i * square_size), (255, 0, 0), 2)

                # Draw square labels pada flattened board
                files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
                ranks = ['1', '2', '6', '5', '4', '3', '2', '1']

                for i in range(8):
                    for j in range(8):
                        square_name = files[j] + ranks[i]
                        if square_name in grid_coords:
                            x, y = grid_coords[square_name]
                            cv2.putText(overlay_image, square_name, (x-15, y+5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                return overlay_image
            
            # Fallback: gambar pada original image jika tidak ada flattened
            elif corners is not None:
                overlay_image = image.copy()
                
                # Draw board corners
                for corner in corners:
                    cv2.circle(overlay_image, tuple(corner.astype(int)), 8, (0, 0, 255), -1)

                # Draw perspective grid lines jika ada corners
                if len(corners) == 4:
                    # Sort corners
                    rect = np.zeros((4, 2), dtype=np.float32)
                    s = corners.sum(axis=1)
                    rect[0] = corners[np.argmin(s)]  # top-left
                    rect[2] = corners[np.argmax(s)]  # bottom-right

                    diff = np.diff(corners, axis=1)
                    rect[1] = corners[np.argmin(diff)]  # top-right
                    rect[3] = corners[np.argmax(diff)]  # bottom-left

                    # Draw grid berdasarkan perspective
                    for i in range(9):
                        # Vertical lines
                        ratio = i / 8.0
                        top_point = (
                            int(rect[0][0] + ratio * (rect[1][0] - rect[0][0])),
                            int(rect[0][1] + ratio * (rect[1][1] - rect[0][1]))
                        )
                        bottom_point = (
                            int(rect[3][0] + ratio * (rect[2][0] - rect[3][0])),
                            int(rect[3][1] + ratio * (rect[2][1] - rect[3][1]))
                        )
                        cv2.line(overlay_image, top_point, bottom_point, (255, 0, 0), 2)
                        
                        # Horizontal lines
                        left_point = (
                            int(rect[0][0] + ratio * (rect[3][0] - rect[0][0])),
                            int(rect[0][1] + ratio * (rect[3][1] - rect[0][1]))
                        )
                        right_point = (
                            int(rect[1][0] + ratio * (rect[2][0] - rect[1][0])),
                            int(rect[1][1] + ratio * (rect[2][1] - rect[1][1]))
                        )
                        cv2.line(overlay_image, left_point, right_point, (255, 0, 0), 2)

                return overlay_image
            else:
                return image

        except Exception as e:
            print(f"Board overlay error: {e}")
            return image

    # ========== ENHANCED DETECTION METHODS ==========
    
    
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
            print(f"Show board grid: {self.show_board_grid}")
            print("Controls: 'q' to quit, 'space' to toggle bbox, 'm' to toggle mode, 'g' to toggle grid, 'b' to toggle board detection")
            
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
                    elif key == ord('g'):
                        self.show_board_grid = not self.show_board_grid
                        print(f"Board grid: {'ON' if self.show_board_grid else 'OFF'}")
                    elif key == ord('b'):
                        self.board_detection_enabled = not self.board_detection_enabled
                        print(f"Board detection: {'ON' if self.board_detection_enabled else 'OFF'}")
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
            cv2.rectangle(overlay, (10, 10), (450, 200), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
            
            # Add text information
            cv2.putText(display_frame, f"Camera: {self.camera_index}", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Mode: {self.detection_mode.upper()}", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_frame, f"BBox: {'ON' if self.show_bbox else 'OFF'}", (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Grid: {'ON' if self.show_board_grid else 'OFF'}", (20, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Board: {'ON' if self.board_detection_enabled else 'OFF'}", (20, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Status flattened board
            flattened_status = "YES" if hasattr(self, 'flattened_board') and self.flattened_board is not None else "NO"
            cv2.putText(display_frame, f"Flattened: {flattened_status}", (20, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (20, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(display_frame, "Q:quit | Space:bbox | M:mode | G:grid | B:board | R:reset", (20, 170), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
            cv2.putText(display_frame, f"Frame: {self.fps_counter}", (20, 190), 
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
    
    def update_detection_settings(self, camera_index=None, mode=None, show_bbox=None, show_board_grid=None, board_detection_enabled=None):
        """Update detection settings during runtime"""
        if camera_index is not None:
            self.camera_index = camera_index
        if mode is not None:
            self.detection_mode = mode
        if show_bbox is not None:
            self.show_bbox = show_bbox
        if show_board_grid is not None:
            self.show_board_grid = show_board_grid
        if board_detection_enabled is not None:
            self.board_detection_enabled = board_detection_enabled
        
        print(f"Settings updated - Camera: {self.camera_index}, Mode: {self.detection_mode}, BBox: {self.show_bbox}, Grid: {self.show_board_grid}, Board: {self.board_detection_enabled}")
    
    def is_detection_active(self):
        """Check if detection is currently running"""
        return self.detection_active
    
    def detect_pieces_realtime(self, image):
        """Enhanced detect pieces with board detection integration"""
        if self.model is None or image is None:
            return image
            
        try:
            input_image = image.copy()
            processed_image = self.crop_to_square(input_image, 720)
            
            if processed_image is None:
                return image
            
            if self.detection_mode == 'clahe':
                processed_image = self.apply_clahe(processed_image)
            
            # Board detection (less frequent for performance)
            board_corners = None
            board_grid_coords = None
            flattened_board = None
            
            if self.board_detection_enabled and self.fps_counter % 10 == 0:  # Every 10th frame
                board_corners, _, board_grid_coords, flattened_board = self.detect_chessboard(processed_image)
                self.board_corners = board_corners
                self.grid_points = board_grid_coords
                self.flattened_board = flattened_board
            else:
                # Use cached board detection
                board_corners = self.board_corners
                board_grid_coords = self.grid_points
                flattened_board = getattr(self, 'flattened_board', None)
            
            # Piece detection - SELALU deteksi pada image original (processed_image)
            display_image = processed_image
            
            if self.fps_counter % 3 == 0:
                results = self.model(processed_image, verbose=False)
                
                if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                    if self.show_bbox:
                        display_image = results[0].plot()
                    else:
                        display_image = processed_image
                    self.last_piece_results = results[0]
                else:
                    display_image = processed_image
                    self.last_piece_results = None
                
                self.last_detection_result = display_image
            else:
                # Use cached results
                if hasattr(self, 'last_detection_result'):
                    display_image = self.last_detection_result
                else:
                    display_image = processed_image
            
            # Logika tampilan berdasarkan mode
            if self.show_board_grid and flattened_board is not None and board_grid_coords is not None:
                # Mode Grid: Tampilkan flattened board dengan grid + overlay bounding box dari original
                
                # 1. Buat flattened board dengan grid
                grid_image = self.draw_chessboard_overlay(
                    flattened_board, board_corners, board_grid_coords, 
                    flattened_board, use_flattened=True
                )
                
                # 2. Jika ada bounding box, overlay ke grid image
                if self.show_bbox and hasattr(self, 'last_piece_results') and self.last_piece_results is not None:
                    # Transform bounding boxes dari original ke flattened coordinate
                    final_image = self._overlay_bbox_on_flattened(grid_image, self.last_piece_results, 
                                                                 board_corners, processed_image.shape)
                else:
                    final_image = grid_image
                    
            else:
                # Mode Normal: Tampilkan original image dengan/tanpa grid perspective
                if self.show_board_grid and board_corners is not None:
                    final_image = self.draw_chessboard_overlay(
                        display_image, board_corners, board_grid_coords, 
                        flattened_board, use_flattened=False
                    )
                else:
                    final_image = display_image
            
            return final_image
                
        except Exception as e:
            print(f"Enhanced detection error: {e}")
            return image
    
    def _overlay_bbox_on_flattened(self, flattened_image, piece_results, board_corners, original_shape):
        """Overlay bounding boxes from original image onto flattened board"""
        try:
            if board_corners is None or piece_results is None:
                return flattened_image
            
            overlay_image = flattened_image.copy()
            
            # Buat homography matrix untuk transform koordinat
            rect = np.zeros((4, 2), dtype=np.float32)
            s = board_corners.sum(axis=1)
            rect[0] = board_corners[np.argmin(s)]  # top-left
            rect[2] = board_corners[np.argmax(s)]  # bottom-right

            diff = np.diff(board_corners, axis=1)
            rect[1] = board_corners[np.argmin(diff)]  # top-right
            rect[3] = board_corners[np.argmax(diff)]  # bottom-left

            # Destination points untuk flattened (720x720)
            dst = np.array([[0, 0], [720, 0], [720, 720], [0, 720]], dtype=np.float32)
            
            # Compute homography matrix
            M = cv2.getPerspectiveTransform(rect, dst)
            
            # Transform setiap bounding box
            for box in piece_results.boxes:
                # Get bounding box coordinates dari original image
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Transform corner points of bounding box
                bbox_corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
                bbox_corners = bbox_corners.reshape(-1, 1, 2)
                
                # Apply homography transformation
                transformed_corners = cv2.perspectiveTransform(bbox_corners, M)
                transformed_corners = transformed_corners.reshape(-1, 2)
                
                # Get new bounding box dari transformed corners
                tx1 = int(np.min(transformed_corners[:, 0]))
                ty1 = int(np.min(transformed_corners[:, 1]))
                tx2 = int(np.max(transformed_corners[:, 0]))
                ty2 = int(np.max(transformed_corners[:, 1]))
                
                # Clamp coordinates ke image bounds
                tx1 = max(0, min(tx1, 720))
                ty1 = max(0, min(ty1, 720))
                tx2 = max(0, min(tx2, 720))
                ty2 = max(0, min(ty2, 720))
                
                # Draw bounding box pada flattened image
                if tx2 > tx1 and ty2 > ty1:  # Valid bounding box
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id] if hasattr(self.model, 'names') else f"Class_{class_id}"
                    
                    # Draw rectangle
                    cv2.rectangle(overlay_image, (tx1, ty1), (tx2, ty2), (0, 255, 0), 2)
                    
                    # Draw label
                    label = f"{class_name}: {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    
                    # Background untuk label
                    cv2.rectangle(overlay_image, (tx1, ty1 - label_size[1] - 10), 
                                (tx1 + label_size[0], ty1), (0, 255, 0), -1)
                    
                    # Text label
                    cv2.putText(overlay_image, label, (tx1, ty1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            return overlay_image
            
        except Exception as e:
            print(f"Bbox overlay error: {e}")
            return flattened_image
    
    # Enhanced web API methods
    def detect_pieces(self, image, mode='raw', show_bbox=True, show_board_grid=True, use_flattened=True):
        """Enhanced detect pieces for web API with board detection"""
        if self.model is None:
            print("YOLO model not loaded")
            return image, None, None, None
            
        try:
            input_image = image.copy()
            processed_image = self.crop_to_square(input_image, 720)
            
            if processed_image is None:
                return image, None, None, None
            
            if mode == 'clahe':
                processed_image = self.apply_clahe(processed_image)
            
            # Board detection
            board_corners, intersections, grid_coords, flattened_board = self.detect_chessboard(processed_image)
            
            # Piece detection - SELALU pada original processed image
            results = self.model(processed_image, verbose=False)
            
            piece_results = None
            if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                piece_results = results[0]
            
            # Tentukan final image berdasarkan mode
            if use_flattened and show_board_grid and flattened_board is not None and grid_coords is not None:
                # Mode Flattened dengan Grid
                
                # 1. Buat flattened board dengan grid
                final_image = self.draw_chessboard_overlay(
                    flattened_board, board_corners, grid_coords, 
                    flattened_board, use_flattened=True
                )
                
                # 2. Overlay bounding boxes jika ada dan diminta
                if show_bbox and piece_results is not None:
                    final_image = self._overlay_bbox_on_flattened(final_image, piece_results, 
                                                                 board_corners, processed_image.shape)
                    
            else:
                # Mode Normal - original image
                if show_bbox and piece_results is not None:
                    final_image = results[0].plot()
                else:
                    final_image = processed_image
                
                # Add board overlay jika diminta
                if show_board_grid and grid_coords is not None:
                    final_image = self.draw_chessboard_overlay(
                        final_image, board_corners, grid_coords, 
                        flattened_board, use_flattened=False
                    )
            
            return final_image, piece_results, board_corners, grid_coords
                
        except Exception as e:
            print(f"Web detection error: {e}")
            return image, None, None, None
    
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

    def get_board_data(self, corners, grid_coords):
        """Extract board detection data"""
        board_data = {
            'corners_detected': corners is not None,
            'corners': corners.tolist() if corners is not None else None,
            'grid_coordinates': grid_coords if grid_coords is not None else {},
            'squares_detected': len(grid_coords) if grid_coords is not None else 0
        }
        return board_data