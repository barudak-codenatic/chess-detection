import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time
import chess
import chess.engine
from sklearn.cluster import DBSCAN
from scipy import ndimage
import pygame
import sys
import os

class ChessBoardDetector:
    def __init__(self):
        self.board_corners = None
        self.grid_points = None

    def detect_board_canny_minimal(self, image):
        """Preprocessing minimal untuk fokus ke deteksi garis"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 30, 90, apertureSize=3)
        return edges

    def detect_lines_hough(self, edges):
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=120)
        horizontal_lines = []
        vertical_lines = []

        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                if abs(theta) < np.pi/12 or abs(theta - np.pi) < np.pi/12:
                    horizontal_lines.append((rho, theta))
                elif abs(theta - np.pi/2) < np.pi/12:
                    vertical_lines.append((rho, theta))

        return horizontal_lines, vertical_lines

    def cluster_lines(self, lines, tolerance=20):
        if not lines:
            return []

        rhos = np.array([line[0] for line in lines]).reshape(-1, 1)
        clustering = DBSCAN(eps=tolerance, min_samples=1).fit(rhos)

        clustered_lines = []
        for cluster_id in set(clustering.labels_):
            cluster_lines = [lines[i] for i in range(len(lines)) if clustering.labels_[i] == cluster_id]
            chosen_line = cluster_lines[0]
            clustered_lines.append(chosen_line)

        return sorted(clustered_lines, key=lambda x: x[0])

    def complete_lines_to_grid(self, lines, is_horizontal=True, image_shape=None):
        """Melengkapi garis menjadi 9 garis untuk membentuk 8 petak"""
        if len(lines) == 0:
            return []
        
        if len(lines) >= 9:
            indices = np.linspace(0, len(lines)-1, 9, dtype=int)
            return [lines[i] for i in indices]
        
        sorted_lines = sorted(lines, key=lambda x: x[0])
        
        if len(sorted_lines) < 2:
            return sorted_lines
        
        distances = []
        for i in range(1, len(sorted_lines)):
            distances.append(abs(sorted_lines[i][0] - sorted_lines[i-1][0]))
        
        if len(distances) == 0:
            return sorted_lines
            
        avg_distance = np.mean(distances)
        completed_lines = sorted_lines.copy()
        
        while len(completed_lines) < 9:
            first_rho = completed_lines[0][0]
            new_rho = first_rho - avg_distance
            
            if new_rho > 0:
                theta = completed_lines[0][1]
                completed_lines.insert(0, (new_rho, theta))
            else:
                break
        
        while len(completed_lines) < 9:
            last_rho = completed_lines[-1][0]
            new_rho = last_rho + avg_distance
            
            max_limit = image_shape[0] if is_horizontal else image_shape[1]
            if new_rho < max_limit:
                theta = completed_lines[-1][1]
                completed_lines.append((new_rho, theta))
            else:
                break
        
        return completed_lines[:9]

    def line_intersections(self, h_lines, v_lines):
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

    def detect_board_corners(self, intersections, image_shape):
        if len(intersections) < 4:
            return None

        points = np.array(intersections)
        min_x, min_y = np.min(points, axis=0)
        max_x, max_y = np.max(points, axis=0)

        def closest_point(target):
            distances = np.linalg.norm(points - target, axis=1)
            return tuple(points[np.argmin(distances)])

        top_left = closest_point((min_x, min_y))
        top_right = closest_point((max_x, min_y))
        bottom_right = closest_point((max_x, max_y))
        bottom_left = closest_point((min_x, max_y))

        corners = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
        return corners

    def apply_homography(self, image, corners):
        if corners is None:
            return None, None

        rect = np.zeros((4, 2), dtype=np.float32)
        s = corners.sum(axis=1)
        rect[0] = corners[np.argmin(s)]
        rect[2] = corners[np.argmax(s)]

        diff = np.diff(corners, axis=1)
        rect[1] = corners[np.argmin(diff)]
        rect[3] = corners[np.argmax(diff)]

        dst = np.array([[0, 0], [400, 0], [400, 400], [0, 400]], dtype=np.float32)

        try:
            M = cv2.getPerspectiveTransform(rect, dst)
            flattened = cv2.warpPerspective(image, M, (400, 400))
            return flattened, M
        except:
            return None, None

    def generate_grid_coordinates(self, size=400):
        coords = {}
        files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        ranks = ['8', '7', '6', '5', '4', '3', '2', '1']

        square_size = size // 8

        for i in range(8):
            for j in range(8):
                square_name = files[j] + ranks[i]
                x = j * square_size + square_size // 2
                y = i * square_size + square_size // 2
                coords[square_name] = (x, y)

        return coords

    def detect_chessboard(self, image):
        """Main chessboard detection method"""
        try:
            edges = self.detect_board_canny_minimal(image)
            h_lines_raw, v_lines_raw = self.detect_lines_hough(edges)
            
            h_lines_clustered = self.cluster_lines(h_lines_raw, tolerance=20)
            v_lines_clustered = self.cluster_lines(v_lines_raw, tolerance=20)
            
            h_lines_complete = self.complete_lines_to_grid(h_lines_clustered, True, image.shape)
            v_lines_complete = self.complete_lines_to_grid(v_lines_clustered, False, image.shape)
            
            if len(h_lines_complete) >= 7 and len(v_lines_complete) >= 7:
                intersections = self.line_intersections(h_lines_complete, v_lines_complete)
                corners = self.detect_board_corners(intersections, image.shape)
                
                if corners is not None:
                    flattened, homography = self.apply_homography(image, corners)
                    if flattened is not None:
                        grid_coords = self.generate_grid_coordinates()
                        return corners, flattened, homography, grid_coords
            
            return None, None, None, None
            
        except Exception as e:
            print(f"Chessboard detection error: {e}")
            return None, None, None, None

class ChessDetectionService:
    def __init__(self, model_path='app/model/best.pt'):
        try:
            self.model = YOLO(model_path)
            print(f"YOLO model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.model = None
        
        # Detection state
        self.detection_active = False
        self.detection_thread = None
        self.camera_index = 0
        self.detection_mode = 'raw'
        self.show_bbox = True
        self.cap = None
        
        # Performance counters
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.skip_frames = 3  # Skip frames for better performance
        
        # Chessboard detection
        self.board_detector = ChessBoardDetector()
        self.board_corners = None
        self.grid_coordinates = None
        self.homography_matrix = None
        self.board_detected = False
        self.flattened_board = None
        
        # Chess board state
        self.current_board = chess.Board()
        self.last_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        self.fen_updated = False
        
        # Enhanced NMS settings
        self.confidence_threshold = 0.7  # Increased for better accuracy
        self.nms_threshold = 0.3  # Strict NMS to prevent duplicates
        
        # Chess visualization
        self.chess_window = None
        self.chess_thread = None
        self.pygame_initialized = False
        
        # Last processed results for efficiency
        self.last_detection_result = None
        self.last_processed_frame = None

    def apply_enhanced_nms(self, detections, iou_threshold=0.25, confidence_threshold=0.7):
        """Enhanced NMS to prevent duplicate detections"""
        if len(detections) == 0:
            return detections
        
        # Filter by confidence
        high_conf_detections = [d for d in detections if d['confidence'] >= confidence_threshold]
        
        if len(high_conf_detections) == 0:
            return []
        
        # Group by class and apply stricter NMS
        class_groups = {}
        for detection in high_conf_detections:
            class_name = detection['class_name']
            if class_name not in class_groups:
                class_groups[class_name] = []
            class_groups[class_name].append(detection)
        
        final_detections = []
        
        for class_name, class_detections in class_groups.items():
            if len(class_detections) == 1:
                final_detections.extend(class_detections)
                continue
            
            # Sort by confidence
            class_detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Apply custom NMS with area consideration
            keep = []
            for i, det in enumerate(class_detections):
                should_keep = True
                bbox1 = det['bbox']
                area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
                
                for j in keep:
                    bbox2 = class_detections[j]['bbox']
                    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
                    
                    # Calculate IoU
                    iou = self.calculate_iou(bbox1, bbox2)
                    
                    # If high overlap, keep the one with higher confidence and reasonable area
                    if iou > iou_threshold:
                        if det['confidence'] <= class_detections[j]['confidence']:
                            should_keep = False
                            break
                        # If areas are very different, might be different pieces
                        area_ratio = min(area1, area2) / max(area1, area2)
                        if area_ratio < 0.3:  # Very different sizes
                            continue
                        should_keep = False
                        break
                
                if should_keep:
                    keep.append(i)
            
            final_detections.extend([class_detections[i] for i in keep])
        
        return final_detections

    def calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union of two bounding boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0

    def crop_to_square(self, image, size=640):
        """Optimized crop to square"""
        try:
            if image is None:
                return None
                
            h, w = image.shape[:2]
            
            if h > w:
                start = (h - w) // 2
                cropped = image[start:start + w, :]
            elif w > h:
                start = (w - h) // 2
                cropped = image[:, start:start + h]
            else:
                cropped = image
            
            return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_LINEAR)
            
        except Exception as e:
            print(f"Crop error: {e}")
            return image

    def apply_clahe(self, image):
        """Lightweight CLAHE enhancement"""
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        except:
            return image

    def map_pieces_to_squares(self, detections):
        """Map detected pieces to chess squares"""
        if not detections or self.grid_coordinates is None:
            return {}
        
        piece_mapping = {}
        
        for detection in detections:
            bbox = detection['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Transform coordinates if we have homography
            if self.homography_matrix is not None:
                # For flattened board coordinates
                transformed_point = np.array([[[center_x, center_y]]], dtype=np.float32)
                try:
                    # Need to transform from original to flattened
                    # This is simplified - in real implementation you'd need proper coordinate mapping
                    pass
                except:
                    pass
            
            # Find closest square (using flattened board coordinates)
            min_distance = float('inf')
            closest_square = None
            
            for square, (sq_x, sq_y) in self.grid_coordinates.items():
                distance = np.sqrt((center_x - sq_x)**2 + (center_y - sq_y)**2)
                if distance < min_distance:
                    min_distance = distance
                    closest_square = square
            
            # Map to square if close enough
            if closest_square and min_distance < 30:  # Stricter threshold
                if closest_square not in piece_mapping or detection['confidence'] > piece_mapping[closest_square]['confidence']:
                    piece_mapping[closest_square] = {
                        'class_name': detection['class_name'],
                        'confidence': detection['confidence']
                    }
        
        return {square: data['class_name'] for square, data in piece_mapping.items()}

    def generate_fen_from_mapping(self, piece_mapping):
        """Generate FEN string from piece mapping"""
        try:
            board = [['' for _ in range(8)] for _ in range(8)]
            
            for square, piece_name in piece_mapping.items():
                file_idx = ord(square[0]) - ord('a')
                rank_idx = 8 - int(square[1])
                
                piece_symbol = self.piece_name_to_fen(piece_name)
                if piece_symbol:
                    board[rank_idx][file_idx] = piece_symbol
            
            fen_rows = []
            for row in board:
                fen_row = ''
                empty_count = 0
                for square in row:
                    if square == '':
                        empty_count += 1
                    else:
                        if empty_count > 0:
                            fen_row += str(empty_count)
                            empty_count = 0
                        fen_row += square
                if empty_count > 0:
                    fen_row += str(empty_count)
                fen_rows.append(fen_row)
            
            fen_position = '/'.join(fen_rows)
            full_fen = f"{fen_position} w KQkq - 0 1"
            
            # Validate FEN
            try:
                chess.Board(full_fen)
                return full_fen
            except:
                return None
                
        except Exception as e:
            print(f"FEN generation error: {e}")
            return None

    def piece_name_to_fen(self, piece_name):
        """Convert piece class name to FEN notation"""
        mapping = {
            'white_pawn': 'P', 'black_pawn': 'p',
            'white_rook': 'R', 'black_rook': 'r',
            'white_knight': 'N', 'black_knight': 'n',
            'white_bishop': 'B', 'black_bishop': 'b',
            'white_queen': 'Q', 'black_queen': 'q',
            'white_king': 'K', 'black_king': 'k'
        }
        return mapping.get(piece_name.lower().replace(' ', '_'), None)

    def init_pygame_chess(self):
        """Initialize pygame for chess visualization"""
        try:
            if not self.pygame_initialized:
                pygame.init()
                self.pygame_initialized = True
            
            self.BOARD_SIZE = 480
            self.SQUARE_SIZE = self.BOARD_SIZE // 8
            self.PANEL_WIDTH = 250
            self.WINDOW_WIDTH = self.BOARD_SIZE + self.PANEL_WIDTH
            self.WINDOW_HEIGHT = self.BOARD_SIZE + 100
            
            self.chess_screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
            pygame.display.set_caption("Chess Board - Live Detection")
            
            self.font = pygame.font.SysFont("arial", 14)
            self.font_large = pygame.font.SysFont("arial", 18, bold=True)
            
            self.LIGHT_SQUARE = (240, 217, 181)
            self.DARK_SQUARE = (181, 136, 99)
            self.HIGHLIGHT_COLOR = (255, 255, 0)
            
            return True
            
        except Exception as e:
            print(f"Pygame initialization error: {e}")
            return False

    def draw_chess_board(self):
        """Draw the chess board with pieces"""
        try:
            if not hasattr(self, 'chess_screen'):
                return
            
            self.chess_screen.fill((40, 40, 40))
            
            # Draw board squares
            for row in range(8):
                for col in range(8):
                    color = self.LIGHT_SQUARE if (row + col) % 2 == 0 else self.DARK_SQUARE
                    rect = pygame.Rect(col * self.SQUARE_SIZE, row * self.SQUARE_SIZE, 
                                     self.SQUARE_SIZE, self.SQUARE_SIZE)
                    pygame.draw.rect(self.chess_screen, color, rect)
            
            # Draw pieces
            self.draw_pieces_on_board()
            
            # Draw coordinates
            self.draw_coordinates()
            
            # Draw info panel
            self.draw_info_panel()
            
            pygame.display.flip()
            
        except Exception as e:
            print(f"Chess board drawing error: {e}")

    def draw_pieces_on_board(self):
        """Draw pieces using Unicode symbols"""
        try:
            piece_symbols = {
                'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
                'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
            }
            
            if self.last_fen:
                try:
                    board = chess.Board(self.last_fen)
                    
                    for square in chess.SQUARES:
                        piece = board.piece_at(square)
                        if piece:
                            file = chess.square_file(square)
                            rank = chess.square_rank(square)
                            
                            x = file * self.SQUARE_SIZE
                            y = (7 - rank) * self.SQUARE_SIZE
                            
                            symbol = piece_symbols.get(piece.symbol(), piece.symbol())
                            color = (255, 255, 255) if piece.color == chess.WHITE else (50, 50, 50)
                            
                            text_surface = pygame.font.Font(None, 48).render(symbol, True, color)
                            text_rect = text_surface.get_rect()
                            text_rect.center = (x + self.SQUARE_SIZE // 2, y + self.SQUARE_SIZE // 2)
                            
                            self.chess_screen.blit(text_surface, text_rect)
                            
                except Exception as e:
                    print(f"Piece drawing error: {e}")
                    
        except Exception as e:
            print(f"Piece rendering error: {e}")

    def draw_coordinates(self):
        """Draw board coordinates"""
        try:
            files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
            ranks = ['8', '7', '6', '5', '4', '3', '2', '1']
            
            # File labels
            for i, file_label in enumerate(files):
                x = i * self.SQUARE_SIZE + self.SQUARE_SIZE // 2
                y = self.BOARD_SIZE + 10
                text = self.font.render(file_label, True, (255, 255, 255))
                text_rect = text.get_rect(center=(x, y))
                self.chess_screen.blit(text, text_rect)
            
            # Rank labels
            for i, rank_label in enumerate(ranks):
                x = -10
                y = i * self.SQUARE_SIZE + self.SQUARE_SIZE // 2
                text = self.font.render(rank_label, True, (255, 255, 255))
                text_rect = text.get_rect(center=(x, y))
                self.chess_screen.blit(text, text_rect)
                
        except Exception as e:
            print(f"Coordinate drawing error: {e}")

    def draw_info_panel(self):
        """Draw information panel"""
        try:
            panel_x = self.BOARD_SIZE + 20
            
            # Background
            panel_rect = pygame.Rect(self.BOARD_SIZE, 0, self.PANEL_WIDTH, self.WINDOW_HEIGHT)
            pygame.draw.rect(self.chess_screen, (25, 25, 25), panel_rect)
            
            # Title
            title = self.font_large.render("Live Chess Detection", True, (255, 255, 255))
            self.chess_screen.blit(title, (panel_x, 20))
            
            y_offset = 60
            
            # Detection status
            board_status = "✓ DETECTED" if self.board_detected else "⚠ SEARCHING"
            board_color = (0, 255, 0) if self.board_detected else (255, 255, 0)
            text = self.font.render(f"Board: {board_status}", True, board_color)
            self.chess_screen.blit(text, (panel_x, y_offset))
            y_offset += 25
            
            # FEN info
            text = self.font.render("Current FEN:", True, (255, 255, 255))
            self.chess_screen.blit(text, (panel_x, y_offset))
            y_offset += 20
            
            if self.last_fen:
                fen_parts = self.last_fen.split(' ')
                # Board position
                rows = fen_parts[0].split('/')
                for row in rows:
                    text = pygame.font.Font(None, 16).render(row, True, (200, 200, 200))
                    self.chess_screen.blit(text, (panel_x, y_offset))
                    y_offset += 18
                
                y_offset += 10
                # Game state
                for i, part in enumerate(fen_parts[1:], 1):
                    label = ["Turn:", "Castling:", "En passant:", "Halfmove:", "Fullmove:"][i-1]
                    text = self.font.render(f"{label} {part}", True, (180, 180, 180))
                    self.chess_screen.blit(text, (panel_x, y_offset))
                    y_offset += 18
            
            y_offset += 20
            
            # Controls
            controls = [
                "Controls:",
                "Q - Quit chess window",
                "R - Reset to start position",
                "S - Save current FEN",
                "Space - Print FEN to console"
            ]
            
            for i, control in enumerate(controls):
                color = (255, 255, 255) if i == 0 else (150, 150, 150)
                weight = True if i == 0 else False
                text = pygame.font.SysFont("arial", 12, bold=weight).render(control, True, color)
                self.chess_screen.blit(text, (panel_x, y_offset))
                y_offset += 16
                
        except Exception as e:
            print(f"Info panel drawing error: {e}")

    def run_chess_window(self):
        """Run the pygame chess window"""
        try:
            if not self.init_pygame_chess():
                return
            
            clock = pygame.time.Clock()
            
            while self.detection_active:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.detection_active = False
                        break
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            self.detection_active = False
                            break
                        elif event.key == pygame.K_r:
                            self.last_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
                            self.current_board = chess.Board()
                            print("Board reset to starting position")
                        elif event.key == pygame.K_s:
                            print(f"FEN saved: {self.last_fen}")
                        elif event.key == pygame.K_SPACE:
                            print(f"Current FEN: {self.last_fen}")
                
                self.draw_chess_board()
                clock.tick(30)
                
        except Exception as e:
            print(f"Chess window error: {e}")
        finally:
            if self.pygame_initialized:
                pygame.quit()
                self.pygame_initialized = False

    def detect_pieces_realtime(self, image):
        """Optimized real-time detection with chessboard and pieces"""
        if self.model is None or image is None:
            return image
            
        try:
            # Crop and preprocess
            processed_image = self.crop_to_square(image, 640)
            if processed_image is None:
                return image
            
            if self.detection_mode == 'clahe':
                processed_image = self.apply_clahe(processed_image)
            
            # Chessboard detection (every 15 frames for performance)
            if self.fps_counter % 15 == 0:
                corners, flattened, homography, grid_coords = self.board_detector.detect_chessboard(processed_image)
                if corners is not None:
                    self.board_corners = corners
                    self.flattened_board = flattened
                    self.homography_matrix = homography
                    self.grid_coordinates = grid_coords
                    self.board_detected = True
                else:
                    self.board_detected = False
            
            # Piece detection (every skip_frames for performance)
            if self.fps_counter % self.skip_frames == 0:
                results = self.model(processed_image, verbose=False, conf=self.confidence_threshold)
                
                if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                    detections = self.get_detection_data(results[0])
                    detections = self.apply_enhanced_nms(detections, 
                                                       iou_threshold=self.nms_threshold,
                                                       confidence_threshold=self.confidence_threshold)
                    
                    # Map pieces to squares and generate FEN
                    if self.grid_coordinates is not None and self.board_detected:
                        piece_mapping = self.map_pieces_to_squares(detections)
                        fen = self.generate_fen_from_mapping(piece_mapping)
                        
                        if fen and fen != self.last_fen:
                            self.last_fen = fen
                            self.fen_updated = True
                            try:
                                self.current_board.set_fen(fen)
                            except Exception as e:
                                print(f"Invalid FEN: {fen}")
                    
                    # Draw detections
                    if self.show_bbox:
                        for detection in detections:
                            bbox = detection['bbox']
                            x1, y1, x2, y2 = map(int, bbox)
                            
                            conf = detection['confidence']
                            if conf > 0.8:
                                color = (0, 255, 0)
                            elif conf > 0.6:
                                color = (255, 255, 0)
                            else:
                                color = (255, 0, 0)
                            
                            cv2.rectangle(processed_image, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(processed_image, f"{detection['class_name']}: {conf:.2f}",
                                      (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                self.last_detection_result = processed_image
                
            return self.last_detection_result if self.last_detection_result is not None else processed_image
                
        except Exception as e:
            print(f"Detection error: {e}")
            return image

    def start_opencv_detection(self, camera_index=0, mode='raw', show_bbox=True):
        """Start detection with both OpenCV and Chess windows"""
        self.camera_index = camera_index
        self.detection_mode = mode
        self.show_bbox = show_bbox
        
        if self.detection_active:
            self.stop_opencv_detection()
        
        self.detection_active = True
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=self._detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        # Start chess window thread
        self.chess_thread = threading.Thread(target=self.run_chess_window)
        self.chess_thread.daemon = True
        self.chess_thread.start()
        
        return True

    def _detection_loop(self):
        """Optimized detection loop"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                print(f"Error: Could not open camera {self.camera_index}")
                self.detection_active = False
                return
            
            # Camera optimization
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            print("Chess Detection Started!")
            print("OpenCV Controls: 'q'=quit, 'space'=toggle bbox, 'm'=toggle mode, 'f'=print FEN")
            print("Chess Window Controls: 'q'=quit, 'r'=reset, 's'=save FEN")
            
            cv2.namedWindow('Chess Detection - Live Feed', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Chess Detection - Live Feed', 800, 800)
            
            frame_count = 0
            start_time = time.time()
            
            while self.detection_active:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                self.fps_counter += 1
                
                # Process frame
                processed_frame = self.detect_pieces_realtime(frame)
                display_frame = self._add_enhanced_overlay(processed_frame)
                
                # Display
                cv2.imshow('Chess Detection - Live Feed', display_frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    self.show_bbox = not self.show_bbox
                    print(f"Bounding boxes: {'ON' if self.show_bbox else 'OFF'}")
                elif key == ord('m'):
                    self.detection_mode = 'clahe' if self.detection_mode == 'raw' else 'raw'
                    print(f"Mode: {self.detection_mode}")
                elif key == ord('f'):
                    print(f"Current FEN: {self.last_fen}")
                elif key == ord('b'):
                    print(f"Board detected: {self.board_detected}")
                
                # FPS monitoring
                frame_count += 1
                if frame_count % 60 == 0:
                    elapsed = time.time() - start_time
                    fps = 60 / elapsed if elapsed > 0 else 0
                    print(f"FPS: {fps:.1f} | Board: {'✓' if self.board_detected else '✗'} | "
                          f"Pieces: {len(self.get_current_pieces())} | FEN: {self.fen_updated}")
                    start_time = time.time()
                    self.fen_updated = False
            
        except Exception as e:
            print(f"Detection loop error: {e}")
        finally:
            try:
                if self.cap:
                    self.cap.release()
                cv2.destroyAllWindows()
            except Exception as e:
                print(f"Cleanup error: {e}")
            
            self.detection_active = False
            print("Detection stopped")

    def _add_enhanced_overlay(self, frame):
        """Add information overlay to frame"""
        try:
            if frame is None:
                return frame
                
            display_frame = frame.copy()
            
            # FPS calculation
            current_time = time.time()
            if hasattr(self, 'last_fps_time'):
                fps = 1.0 / (current_time - self.last_fps_time) if (current_time - self.last_fps_time) > 0 else 0
            else:
                fps = 0
            self.last_fps_time = current_time
            
            # Overlay background
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (10, 10), (500, 150), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
            
            # Status
            board_status = "DETECTED" if self.board_detected else "SEARCHING"
            board_color = (0, 255, 0) if self.board_detected else (0, 255, 255)
            
            # Info text
            cv2.putText(display_frame, f"Camera: {self.camera_index} | Mode: {self.detection_mode.upper()}", 
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"FPS: {fps:.1f} | BBox: {'ON' if self.show_bbox else 'OFF'}", 
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(display_frame, f"Board: {board_status} | Pieces: {len(self.get_current_pieces())}", 
                       (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, board_color, 2)
            cv2.putText(display_frame, f"Confidence: {self.confidence_threshold:.1f} | NMS: {self.nms_threshold:.1f}", 
                       (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, "Controls: Q=quit | Space=bbox | M=mode | F=fen | B=board", 
                       (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(display_frame, f"Frame: {self.fps_counter}", 
                       (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            
            # Draw chessboard corners
            if self.board_corners is not None:
                for i, corner in enumerate(self.board_corners):
                    cv2.circle(display_frame, tuple(corner.astype(int)), 6, (0, 0, 255), -1)
                    cv2.putText(display_frame, str(i), tuple((corner + 10).astype(int)), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            return display_frame
            
        except Exception as e:
            print(f"Overlay error: {e}")
            return frame

    def get_current_pieces(self):
        """Get currently detected pieces"""
        if not hasattr(self, 'last_fen') or not self.last_fen:
            return []
        
        try:
            board = chess.Board(self.last_fen)
            pieces = []
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece:
                    pieces.append(piece.symbol())
            return pieces
        except:
            return []

    def stop_opencv_detection(self):
        """Stop all detection processes"""
        print("Stopping detection...")
        self.detection_active = False
        
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=3)
        
        if self.chess_thread and self.chess_thread.is_alive():
            self.chess_thread.join(timeout=3)
        
        try:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            if self.pygame_initialized:
                pygame.quit()
                self.pygame_initialized = False
        except Exception as e:
            print(f"Cleanup error: {e}")
        
        print("All detection processes stopped")

    # Keep existing API methods for web compatibility
    def detect_pieces(self, image, mode='raw', show_bbox=True):
        """Detect pieces for web API"""
        if self.model is None:
            return image, None
            
        try:
            processed_image = self.crop_to_square(image, 640)
            if processed_image is None:
                return image, None
            
            if mode == 'clahe':
                processed_image = self.apply_clahe(processed_image)
            
            results = self.model(processed_image, verbose=False, conf=self.confidence_threshold)
            
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

    def update_detection_settings(self, camera_index=None, mode=None, show_bbox=None, 
                                confidence_threshold=None, nms_threshold=None):
        """Update detection settings"""
        if camera_index is not None:
            self.camera_index = camera_index
        if mode is not None:
            self.detection_mode = mode
        if show_bbox is not None:
            self.show_bbox = show_bbox
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
        if nms_threshold is not None:
            self.nms_threshold = nms_threshold

    def is_detection_active(self):
        return self.detection_active

    def get_current_fen(self):
        return self.last_fen

    def get_current_board(self):
        return self.current_board

    def get_detection_stats(self):
        return {
            'board_detected': self.board_detected,
            'current_fen': self.last_fen,
            'detection_active': self.detection_active,
            'camera_index': self.camera_index,
            'detection_mode': self.detection_mode,
            'confidence_threshold': self.confidence_threshold,
            'nms_threshold': self.nms_threshold,
            'pieces_count': len(self.get_current_pieces())
        }