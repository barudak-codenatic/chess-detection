import pygame
import chess
import chess.engine
import math

# ==== KONFIGURASI UI ====
BOARD_W, BOARD_H = 640, 640
PANEL_W = 220                        # panel kanan untuk info
WIDTH, HEIGHT = BOARD_W + PANEL_W, BOARD_H
SQUARE_SIZE = BOARD_W // 8
FPS = 30

STOCKFISH_PATH = "./stockfish.exe"   # ganti sesuai lokasi stockfish kamu
ENGINE_DEPTH = 12                    # kedalaman analisis
EVAL_CLAMP = 1000                    # clamp cp untuk tampilan bar (-10 .. +10 pion)

# ==== INIT PYGAME ====
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Chess with Stockfish: Evaluation, Grading & Accuracy")
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 18)
font_small = pygame.font.SysFont("consolas", 14)
font_big = pygame.font.SysFont("consolas", 22, bold=True)

# ==== LOAD BIDAK (Windows-safe naming) ====
pieces_paths = {
    "P": "pawn_w.png",
    "p": "pawn_b.png",
    "N": "knight_w.png",
    "n": "knight_b.png",
    "B": "bishop_w.png",
    "b": "bishop_b.png",
    "R": "rook_w.png",
    "r": "rook_b.png",
    "Q": "queen_w.png",
    "q": "queen_b.png",
    "K": "king_w.png",
    "k": "king_b.png",
}
pieces = {}
for key, filename in pieces_paths.items():
    img = pygame.image.load(f"assets/{filename}")
    pieces[key] = pygame.transform.smoothscale(img, (SQUARE_SIZE, SQUARE_SIZE))

# ==== CHESS & ENGINE ====
board = chess.Board()
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

# ==== STATE ====
selected_square = None
running = True
best_move_suggestion = None          # saran best move utk posisi saat ini
last_grade_text = ""                 # penilaian langkah terakhir
last_cp_loss = None                  # berapa kehilangan cp dibanding best
current_eval_cp = 0                  # evaluasi posisi sekarang (cp dari perspektif putih)
move_logs = []                       # list log langkah: (ply, san, grade, loss, eval_after)

# statistik akurasi per warna
stats = {
    chess.WHITE: {"moves": 0, "acc_sum": 0.0},
    chess.BLACK: {"moves": 0, "acc_sum": 0.0},
}

# ==== UTIL ENGINE/EVAL ====
def score_to_cp_white(score):
    """
    Ubah score python-chess ke centipawn dari perspektif putih.
    mate_score besar agar mate >> cp
    """
    return score.white().score(mate_score=100000)

def analyse_position(bd):
    """
    Kembalikan (eval_cp_from_white, pv_best_move) untuk posisi bd.
    """
    info = engine.analyse(bd, chess.engine.Limit(depth=ENGINE_DEPTH))
    eval_cp = score_to_cp_white(info["score"])
    pv = info.get("pv", [])
    mv = pv[0] if pv else None
    return eval_cp, mv

def eval_after_move(bd, move):
    """
    Dorong move pada board sementara, analisis, lalu pop. Return eval_cp_after.
    """
    bd.push(move)
    info = engine.analyse(bd, chess.engine.Limit(depth=ENGINE_DEPTH))
    eval_cp_after = score_to_cp_white(info["score"])
    bd.pop()
    return eval_cp_after

def classify_move(cp_loss):
    """
    Klasifikasi langkah dari kehilangan CP (loss terhadap best line sisi yg bergerak).
    Ambang mirip ‘rasa’ chess.com (heuristik).
    """
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

def estimate_accuracy_from_loss(cp_loss):
    """
    Estimasi akurasi % dari kehilangan CP.
    Pakai fungsi logistic halus agar skornya natural (0..100).
    """
    loss = max(0.0, float(abs(cp_loss)))
    # parameter skala (semakin kecil => lebih ‘galak’)
    k = 170.0
    # logistic: 100/(1+(loss/k)^0.75)
    acc = 100.0 / (1.0 + (loss / k) ** 0.75)
    return max(0.0, min(100.0, acc))

def uci_to_san_safe(bd, move):
    try:
        return bd.san(move)
    except Exception:
        return move.uci()

def algebra(square):
    return chess.square_name(square)

# ==== RENDERING ====
def draw_board():
    colors = [pygame.Color(240, 217, 181), pygame.Color(181, 136, 99)]  # klasik
    for r in range(8):
        for c in range(8):
            color = colors[(r + c) % 2]
            pygame.draw.rect(
                screen,
                color,
                pygame.Rect(c * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE),
            )

def draw_selection_highlight(square):
    if square is None:
        return
    r = 7 - (square // 8)
    c = square % 8
    pygame.draw.rect(
        screen,
        (50, 200, 50),
        pygame.Rect(c * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE),
        4,
        border_radius=5,
    )

def draw_pieces():
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - (square // 8)
            col = square % 8
            screen.blit(
                pieces[piece.symbol()],
                pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE),
            )

def draw_best_move_highlight(move):
    if not move:
        return
    from_square = move.from_square
    to_square = move.to_square
    r1, c1 = 7 - (from_square // 8), from_square % 8
    r2, c2 = 7 - (to_square // 8), to_square % 8
    pygame.draw.rect(
        screen, (0, 255, 0), pygame.Rect(c1 * SQUARE_SIZE, r1 * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 4, border_radius=6
    )
    pygame.draw.rect(
        screen, (255, 0, 0), pygame.Rect(c2 * SQUARE_SIZE, r2 * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 4, border_radius=6
    )

def draw_eval_bar(eval_cp):
    """
    Bar vertikal di sisi kanan papan (dalam panel).
    eval_cp > 0 menguntungkan putih, < 0 menguntungkan hitam.
    """
    x0 = BOARD_W + 20
    y0 = 20
    bar_w = PANEL_W - 40
    bar_h = HEIGHT - 40

    # frame
    pygame.draw.rect(screen, (220, 220, 220), pygame.Rect(x0, y0, bar_w, bar_h), border_radius=8)
    # garis tengah
    center_y = y0 + bar_h // 2
    pygame.draw.line(screen, (120, 120, 120), (x0 + 5, center_y), (x0 + bar_w - 5, center_y), 1)

    # normalisasi -EVAL_CLAMP..+EVAL_CLAMP -> 0..1 (1 = atas/putih)
    v = max(-EVAL_CLAMP, min(EVAL_CLAMP, eval_cp))
    frac = 0.5 - (v / (2 * EVAL_CLAMP))  # cp positif => putih unggul => isi dari atas
    frac = max(0.0, min(1.0, frac))

    # isi: bagian atas (putih) dan bawah (hitam)
    white_h = int(bar_h * frac)
    black_h = bar_h - white_h

    # putih di atas (cerah)
    pygame.draw.rect(screen, (245, 245, 245), pygame.Rect(x0 + 2, y0 + 2, bar_w - 4, white_h - 2), border_radius=6)
    # hitam di bawah (gelap)
    pygame.draw.rect(screen, (30, 30, 30), pygame.Rect(x0 + 2, y0 + white_h, bar_w - 4, black_h - 2), border_radius=6)

    # label angka eval
    if abs(eval_cp) >= 99999:
        eval_txt = "MATE"
    else:
        eval_txt = f"{eval_cp/100:.2f}"
    label = font_big.render(f"Eval: {eval_txt}", True, (20, 20, 20))
    screen.blit(label, (x0, 10))

def draw_panel_text():
    # judul
    title = font_big.render("Analysis", True, (20, 20, 20))
    screen.blit(title, (BOARD_W + 20, HEIGHT - 210))

    # penilaian langkah terakhir
    g = last_grade_text or "-"
    if last_cp_loss is None:
        loss_txt = "-"
    else:
        loss_txt = f"{last_cp_loss:.0f} cp"
    txt1 = font.render(f"Last move: {g}", True, (20, 20, 20))
    txt2 = font_small.render(f"Loss vs best: {loss_txt}", True, (60, 60, 60))
    screen.blit(txt1, (BOARD_W + 20, HEIGHT - 180))
    screen.blit(txt2, (BOARD_W + 20, HEIGHT - 156))

    # akurasi per warna
    def acc_str(color):
        m = stats[color]["moves"]
        s = stats[color]["acc_sum"]
        return 0.0 if m == 0 else s / m

    acc_w = acc_str(chess.WHITE)
    acc_b = acc_str(chess.BLACK)
    aw = font.render(f"White Acc: {acc_w:5.1f}%", True, (20, 20, 20))
    ab = font.render(f"Black Acc: {acc_b:5.1f}%", True, (20, 20, 20))
    screen.blit(aw, (BOARD_W + 20, HEIGHT - 128))
    screen.blit(ab, (BOARD_W + 20, HEIGHT - 104))

    # saran best move saat ini
    bm_txt = "-"
    if best_move_suggestion:
        bm_txt = f"{uci_to_san_safe(board, best_move_suggestion)}"
    bm = font.render(f"Best: {bm_txt}", True, (20, 20, 20))
    screen.blit(bm, (BOARD_W + 20, HEIGHT - 76))

    # status game
    if board.is_game_over():
        res = board.result()
        over = font_big.render(f"Game Over: {res}", True, (160, 0, 0))
        screen.blit(over, (BOARD_W + 20, HEIGHT - 48))

def get_square_from_mouse(pos):
    x, y = pos
    if x >= BOARD_W or y >= BOARD_H:
        return None
    col = x // SQUARE_SIZE
    row = 7 - (y // SQUARE_SIZE)
    return chess.square(col, row)

# ==== PRE-COMPUTE SUGGESTION & EVAL (START) ====
current_eval_cp, best_move_suggestion = analyse_position(board)

# ==== MAIN LOOP ====
while running:
    clock.tick(FPS)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            sq = get_square_from_mouse(pygame.mouse.get_pos())
            if sq is None:
                selected_square = None
                continue
            if selected_square is None:
                # pilih kotak asal
                if board.piece_at(sq) and (board.piece_at(sq).color == board.turn):
                    selected_square = sq
            else:
                # coba gerakkan
                move = chess.Move(selected_square, sq)
                if move in board.legal_moves:
                    # --- ANALISIS PENILAIAN SEBELUM PUSH ---
                    side = board.turn
                    # best line engine utk side yg bergerak
                    _, engine_best_move = analyse_position(board)

                    # eval setelah langkah terbaik engine (baseline)
                    if engine_best_move:
                        eval_if_best = eval_after_move(board, engine_best_move)
                    else:
                        # fallback jika tak ada pv
                        eval_if_best = current_eval_cp

                    # eval setelah langkah pemain
                    eval_after_player = eval_after_move(board, move)

                    # kehilangan cp relatif terhadap best (positif = lebih buruk dari best)
                    cp_loss = (eval_if_best - eval_after_player)
                    last_cp_loss = cp_loss
                    grade = classify_move(cp_loss)
                    last_grade_text = grade

                    # update akurasi
                    acc = estimate_accuracy_from_loss(cp_loss)
                    stats[side]["moves"] += 1
                    stats[side]["acc_sum"] += acc

                    # benar-benar jalankan langkah
                    board.push(move)

                    # log
                    move_logs.append((
                        board.fullmove_number if side == chess.BLACK else board.fullmove_number,
                        uci_to_san_safe(board, move), grade, cp_loss, eval_after_player
                    ))
                    if len(move_logs) > 12:
                        move_logs.pop(0)

                    # --- ANALISIS SETELAH PUSH (untuk tampilan berikutnya) ---
                    current_eval_cp, best_move_suggestion = analyse_position(board)
                # reset seleksi
                selected_square = None

    # === RENDER ===
    # papan + bidak
    draw_board()
    draw_selection_highlight(selected_square)
    draw_pieces()
    # highlight best move saat ini
    draw_best_move_highlight(best_move_suggestion)
    # panel kanan
    draw_eval_bar(current_eval_cp)
    draw_panel_text()

    pygame.display.flip()

pygame.quit()
engine.quit()
