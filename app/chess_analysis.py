import pygame
import chess
import chess.engine
import threading
import time
import queue
from typing import Optional, Tuple, Dict, Any

class ChessAnalysisService:
    def __init__(self, stockfish_path):
        # ==== KONFIGURASI UI ====
        self.BOARD_W, self.BOARD_H = 640, 640
        self.PANEL_W = 220
        self.WIDTH, self.HEIGHT = self.BOARD_W + self.PANEL_W, self.BOARD_H
        self.SQUARE_SIZE = self.BOARD_W // 8
        self.FPS = 30
        
        self.STOCKFISH_PATH = stockfish_path
        self.ENGINE_DEPTH = 12
        self.EVAL_CLAMP = 1000
        
        # State
        self.analysis_active = False
        self.analysis_thread = None
        self.current_fen = None
        self.board = chess.Board()
        self.engine = None
        self.engine_lock = threading.Lock()  # Lock untuk engine access
        
        # Analysis data
        self.selected_square = None
        self.best_move_suggestion = None
        self.last_grade_text = ""
        self.last_cp_loss = None
        self.current_eval_cp = 0
        self.move_logs = []
        
        # Thread safety
        self.analysis_queue = queue.Queue()
        self.engine_busy = False
        
        # Statistics
        self.stats = {
            chess.WHITE: {"moves": 0, "acc_sum": 0.0},
            chess.BLACK: {"moves": 0, "acc_sum": 0.0},
        }
        
        # Pygame objects (will be initialized in thread)
        self.screen = None
        self.clock = None
        self.font = None
        self.font_small = None
        self.font_big = None
        self.pieces = {}

    def _init_pygame(self):
        """Initialize pygame in the analysis thread"""
        try:
            pygame.init()
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("ChessMon Analysis - Stockfish Evaluation")
            self.clock = pygame.time.Clock()
            
            self.font = pygame.font.SysFont("consolas", 18)
            self.font_small = pygame.font.SysFont("consolas", 14)
            self.font_big = pygame.font.SysFont("consolas", 22, bold=True)
            
            # Load piece images
            self._load_pieces()
            
            return True
        except Exception as e:
            print(f"Failed to initialize pygame: {e}")
            return False

    def _load_pieces(self):
        """Load chess piece images"""
        pieces_paths = {
            "P": "pawn_w.png", "p": "pawn_b.png",
            "N": "knight_w.png", "n": "knight_b.png", 
            "B": "bishop_w.png", "b": "bishop_b.png",
            "R": "rook_w.png", "r": "rook_b.png",
            "Q": "queen_w.png", "q": "queen_b.png",
            "K": "king_w.png", "k": "king_b.png",
        }
        
        self.pieces = {}
        for key, filename in pieces_paths.items():
            try:
                img = pygame.image.load(f"app/assets/{filename}")
                self.pieces[key] = pygame.transform.smoothscale(img, (self.SQUARE_SIZE, self.SQUARE_SIZE))
            except Exception as e:
                print(f"Warning: Could not load piece image {filename}: {e}")
                # Create a colored rectangle as fallback
                surf = pygame.Surface((self.SQUARE_SIZE, self.SQUARE_SIZE))
                color = (255, 255, 255) if key.isupper() else (0, 0, 0)
                surf.fill(color)
                self.pieces[key] = surf

    def _init_engine(self):
        """Initialize Stockfish engine with better error handling"""
        try:
            with self.engine_lock:
                if self.engine:
                    try:
                        self.engine.quit()
                    except:
                        pass
                
                # Try multiple paths
                paths_to_try = [
                    "app/engine/stockfish.exe",
                    "./stockfish.exe", 
                    "stockfish.exe",
                    "stockfish"
                ]
                
                for path in paths_to_try:
                    try:
                        print(f"Trying Stockfish path: {path}")
                        self.engine = chess.engine.SimpleEngine.popen_uci(path)
                        print(f"Stockfish engine initialized from {path}")
                        return True
                    except Exception as e:
                        print(f"Failed with path {path}: {e}")
                        continue
                
                print("Failed to initialize Stockfish engine with any path")
                return False
                
        except Exception as e:
            print(f"Engine initialization error: {e}")
            return False

    def start_analysis(self, initial_fen=None):
        """Start chess analysis window"""
        if self.analysis_active:
            print("Analysis already running")
            return False
            
        self.current_fen = initial_fen
        self.analysis_active = True
        self.analysis_thread = threading.Thread(target=self._analysis_loop)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
        
        return True

    def stop_analysis(self):
        """Stop chess analysis window"""
        print("Stopping chess analysis...")
        self.analysis_active = False
        
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=3)
        
        with self.engine_lock:
            if self.engine:
                try:
                    self.engine.quit()
                    self.engine = None
                except Exception as e:
                    print(f"Engine cleanup error: {e}")
        
        print("Chess analysis stopped")

    def update_fen(self, fen_string):
        """Update board position from FEN string"""
        if not fen_string:
            return False
            
        try:
            new_board = chess.Board(fen_string)  # Test FEN validity
            self.board = new_board
            self.current_fen = fen_string
            
            # Queue analysis update instead of direct call
            if self.analysis_active:
                try:
                    self.analysis_queue.put(('update_analysis', None), block=False)
                except queue.Full:
                    pass  # Skip if queue is full
            
            return True
        except Exception as e:
            print(f"Invalid FEN string: {e}")
            return False

    def _update_analysis(self):
        """Update analysis for current position - thread safe"""
        if self.engine_busy:
            return
            
        try:
            with self.engine_lock:
                if self.engine and not self.engine_busy:
                    self.engine_busy = True
                    self.current_eval_cp, self.best_move_suggestion = self._analyse_position(self.board)
                    self.engine_busy = False
        except Exception as e:
            print(f"Analysis update error: {e}")
            self.engine_busy = False

    def _analysis_loop(self):
        """Main analysis loop"""
        try:
            # Initialize pygame
            if not self._init_pygame():
                self.analysis_active = False
                return
            
            # Initialize engine
            if not self._init_engine():
                print("Warning: Running without engine analysis")
                # Continue without engine
            
            # Set initial position
            if self.current_fen:
                self.update_fen(self.current_fen)
            else:
                self.board = chess.Board()
            
            # Initial analysis (if engine available)
            if self.engine:
                self._update_analysis()
            
            print("Chess analysis window started")
            print("Controls: Click to move pieces, Close window to exit")
            
            # Main loop
            while self.analysis_active:
                self.clock.tick(self.FPS)
                
                # Process analysis queue
                try:
                    while not self.analysis_queue.empty():
                        action, data = self.analysis_queue.get_nowait()
                        if action == 'update_analysis':
                            self._update_analysis()
                except queue.Empty:
                    pass
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.analysis_active = False
                        break
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        self._handle_mouse_click(pygame.mouse.get_pos())
                
                # Render
                self._render()
                pygame.display.flip()
                
        except Exception as e:
            print(f"Analysis loop error: {e}")
        finally:
            try:
                pygame.quit()
            except Exception as e:
                print(f"Pygame cleanup error: {e}")
            
            self.analysis_active = False

    def _handle_mouse_click(self, pos):
        """Handle mouse click for piece movement"""
        sq = self._get_square_from_mouse(pos)
        if sq is None:
            self.selected_square = None
            return
            
        if self.selected_square is None:
            # Select piece
            if self.board.piece_at(sq) and (self.board.piece_at(sq).color == self.board.turn):
                self.selected_square = sq
        else:
            # Try to move
            move = chess.Move(self.selected_square, sq)
            if move in self.board.legal_moves:
                self._execute_move(move)
            self.selected_square = None

    def _execute_move(self, move):
        """Execute move and update analysis - with better error handling"""
        try:
            side = self.board.turn
            
            # Analyze move quality (only if engine available and not busy)
            if self.engine and not self.engine_busy:
                try:
                    with self.engine_lock:
                        if self.engine:
                            self.engine_busy = True
                            
                            # Get current best move
                            _, engine_best_move = self._analyse_position(self.board)
                            
                            if engine_best_move:
                                eval_if_best = self._eval_after_move(self.board, engine_best_move)
                            else:
                                eval_if_best = self.current_eval_cp
                            
                            eval_after_player = self._eval_after_move(self.board, move)
                            cp_loss = eval_if_best - eval_after_player
                            self.last_cp_loss = cp_loss
                            grade = self._classify_move(cp_loss)
                            self.last_grade_text = grade
                            
                            # Update accuracy stats
                            acc = self._estimate_accuracy_from_loss(cp_loss)
                            self.stats[side]["moves"] += 1
                            self.stats[side]["acc_sum"] += acc
                            
                            # Log move
                            self.move_logs.append((
                                self.board.fullmove_number if side == chess.BLACK else self.board.fullmove_number,
                                self._uci_to_san_safe(self.board, move), grade, cp_loss, eval_after_player
                            ))
                            if len(self.move_logs) > 12:
                                self.move_logs.pop(0)
                            
                            self.engine_busy = False
                            
                except Exception as e:
                    print(f"Move analysis error: {e}")
                    self.engine_busy = False
            
            # Execute move
            self.board.push(move)
            
            # Queue analysis update for new position
            try:
                self.analysis_queue.put(('update_analysis', None), block=False)
            except queue.Full:
                pass
            
        except Exception as e:
            print(f"Move execution error: {e}")
            self.engine_busy = False

    def _render(self):
        """Render the chess analysis window"""
        # Clear screen
        self.screen.fill((250, 250, 250))
        
        # Draw board
        self._draw_board()
        self._draw_selection_highlight(self.selected_square)
        self._draw_pieces()
        self._draw_best_move_highlight(self.best_move_suggestion)
        
        # Draw panel
        self._draw_eval_bar(self.current_eval_cp)
        self._draw_panel_text()

    def _draw_board(self):
        """Draw chessboard"""
        colors = [pygame.Color(240, 217, 181), pygame.Color(181, 136, 99)]
        for r in range(8):
            for c in range(8):
                color = colors[(r + c) % 2]
                pygame.draw.rect(
                    self.screen,
                    color,
                    pygame.Rect(c * self.SQUARE_SIZE, r * self.SQUARE_SIZE, self.SQUARE_SIZE, self.SQUARE_SIZE),
                )

    def _draw_selection_highlight(self, square):
        """Draw selection highlight"""
        if square is None:
            return
        r = 7 - (square // 8)
        c = square % 8
        pygame.draw.rect(
            self.screen,
            (50, 200, 50),
            pygame.Rect(c * self.SQUARE_SIZE, r * self.SQUARE_SIZE, self.SQUARE_SIZE, self.SQUARE_SIZE),
            4,
            border_radius=5,
        )

    def _draw_pieces(self):
        """Draw chess pieces"""
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                row = 7 - (square // 8)
                col = square % 8
                self.screen.blit(
                    self.pieces[piece.symbol()],
                    pygame.Rect(col * self.SQUARE_SIZE, row * self.SQUARE_SIZE, self.SQUARE_SIZE, self.SQUARE_SIZE),
                )

    def _draw_best_move_highlight(self, move):
        """Draw best move highlight"""
        if not move:
            return
        from_square = move.from_square
        to_square = move.to_square
        r1, c1 = 7 - (from_square // 8), from_square % 8
        r2, c2 = 7 - (to_square // 8), to_square % 8
        pygame.draw.rect(
            self.screen, (0, 255, 0), 
            pygame.Rect(c1 * self.SQUARE_SIZE, r1 * self.SQUARE_SIZE, self.SQUARE_SIZE, self.SQUARE_SIZE), 
            4, border_radius=6
        )
        pygame.draw.rect(
            self.screen, (255, 0, 0), 
            pygame.Rect(c2 * self.SQUARE_SIZE, r2 * self.SQUARE_SIZE, self.SQUARE_SIZE, self.SQUARE_SIZE), 
            4, border_radius=6
        )

    def _draw_eval_bar(self, eval_cp):
        """Draw evaluation bar"""
        x0 = self.BOARD_W + 20
        y0 = 20
        bar_w = self.PANEL_W - 40
        bar_h = self.HEIGHT - 40

        pygame.draw.rect(self.screen, (220, 220, 220), pygame.Rect(x0, y0, bar_w, bar_h), border_radius=8)
        center_y = y0 + bar_h // 2
        pygame.draw.line(self.screen, (120, 120, 120), (x0 + 5, center_y), (x0 + bar_w - 5, center_y), 1)

        v = max(-self.EVAL_CLAMP, min(self.EVAL_CLAMP, eval_cp))
        frac = 0.5 - (v / (2 * self.EVAL_CLAMP))
        frac = max(0.0, min(1.0, frac))

        white_h = int(bar_h * frac)
        black_h = bar_h - white_h

        pygame.draw.rect(self.screen, (245, 245, 245), pygame.Rect(x0 + 2, y0 + 2, bar_w - 4, white_h - 2), border_radius=6)
        pygame.draw.rect(self.screen, (30, 30, 30), pygame.Rect(x0 + 2, y0 + white_h, bar_w - 4, black_h - 2), border_radius=6)

        if abs(eval_cp) >= 99999:
            eval_txt = "MATE"
        else:
            eval_txt = f"{eval_cp/100:.2f}"
        label = self.font_big.render(f"Eval: {eval_txt}", True, (20, 20, 20))
        self.screen.blit(label, (x0, 10))

    def _draw_panel_text(self):
        """Draw analysis panel text"""
        title = self.font_big.render("Analysis", True, (20, 20, 20))
        self.screen.blit(title, (self.BOARD_W + 20, self.HEIGHT - 210))

        # Engine status
        engine_status = "READY" if self.engine and not self.engine_busy else "NO ENGINE" if not self.engine else "BUSY"
        engine_color = (0, 200, 0) if self.engine and not self.engine_busy else (200, 0, 0) if not self.engine else (200, 200, 0)
        status_text = self.font_small.render(f"Engine: {engine_status}", True, engine_color)
        self.screen.blit(status_text, (self.BOARD_W + 20, self.HEIGHT - 240))

        g = self.last_grade_text or "-"
        if self.last_cp_loss is None:
            loss_txt = "-"
        else:
            loss_txt = f"{self.last_cp_loss:.0f} cp"
        txt1 = self.font.render(f"Last move: {g}", True, (20, 20, 20))
        txt2 = self.font_small.render(f"Loss vs best: {loss_txt}", True, (60, 60, 60))
        self.screen.blit(txt1, (self.BOARD_W + 20, self.HEIGHT - 180))
        self.screen.blit(txt2, (self.BOARD_W + 20, self.HEIGHT - 156))

        def acc_str(color):
            m = self.stats[color]["moves"]
            s = self.stats[color]["acc_sum"]
            return 0.0 if m == 0 else s / m

        acc_w = acc_str(chess.WHITE)
        acc_b = acc_str(chess.BLACK)
        aw = self.font.render(f"White Acc: {acc_w:5.1f}%", True, (20, 20, 20))
        ab = self.font.render(f"Black Acc: {acc_b:5.1f}%", True, (20, 20, 20))
        self.screen.blit(aw, (self.BOARD_W + 20, self.HEIGHT - 128))
        self.screen.blit(ab, (self.BOARD_W + 20, self.HEIGHT - 104))

        bm_txt = "-"
        if self.best_move_suggestion:
            bm_txt = f"{self._uci_to_san_safe(self.board, self.best_move_suggestion)}"
        bm = self.font.render(f"Best: {bm_txt}", True, (20, 20, 20))
        self.screen.blit(bm, (self.BOARD_W + 20, self.HEIGHT - 76))

        if self.board.is_game_over():
            res = self.board.result()
            over = self.font_big.render(f"Game Over: {res}", True, (160, 0, 0))
            self.screen.blit(over, (self.BOARD_W + 20, self.HEIGHT - 48))

    # Helper methods - dengan better error handling
    def _get_square_from_mouse(self, pos):
        """Get chess square from mouse position"""
        x, y = pos
        if x >= self.BOARD_W or y >= self.BOARD_H:
            return None
        col = x // self.SQUARE_SIZE
        row = 7 - (y // self.SQUARE_SIZE)
        return chess.square(col, row)

    def _score_to_cp_white(self, score):
        """Convert score to centipawn from white perspective"""
        try:
            return score.white().score(mate_score=100000)
        except:
            return 0

    def _analyse_position(self, bd):
        """Analyze position and return evaluation and best move"""
        try:
            if not self.engine:
                return 0, None
                
            info = self.engine.analyse(bd, chess.engine.Limit(depth=self.ENGINE_DEPTH))
            eval_cp = self._score_to_cp_white(info["score"])
            pv = info.get("pv", [])
            mv = pv[0] if pv else None
            return eval_cp, mv
        except Exception as e:
            print(f"Position analysis error: {e}")
            return 0, None

    def _eval_after_move(self, bd, move):
        """Get evaluation after making a move"""
        try:
            if not self.engine:
                return 0
                
            bd.push(move)
            info = self.engine.analyse(bd, chess.engine.Limit(depth=self.ENGINE_DEPTH))
            eval_cp_after = self._score_to_cp_white(info["score"])
            bd.pop()
            return eval_cp_after
        except Exception as e:
            print(f"Move evaluation error: {e}")
            if bd.move_stack:  # Ensure we pop if we pushed
                bd.pop()
            return 0

    def _classify_move(self, cp_loss):
        """Classify move quality based on CP loss"""
        loss = abs(cp_loss)
        if loss <= 15:
            return "Best"
        elif loss <= 35:
            return "Excellent"
        elif loss <= 80:
            return "Good"
        elif loss <= 150:
            return "Inaccuracy"
        elif loss <= 400:
            return "Mistake"
        else:
            return "Blunder"

    def _estimate_accuracy_from_loss(self, cp_loss):
        """Estimate accuracy percentage from CP loss"""
        loss = max(0.0, float(abs(cp_loss)))
        k = 170.0
        acc = 100.0 / (1.0 + (loss / k) ** 0.75)
        return max(0.0, min(100.0, acc))

    def _uci_to_san_safe(self, bd, move):
        """Convert UCI move to SAN safely"""
        try:
            return bd.san(move)
        except Exception:
            try:
                return move.uci()
            except:
                return str(move)

    def is_analysis_active(self):
        """Check if analysis window is active"""
        return self.analysis_active