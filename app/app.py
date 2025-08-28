from datetime import datetime
import os
from flask import Flask, jsonify
from flask_login import LoginManager
from flask_socketio import SocketIO, emit, join_room, leave_room
from config import Config
from models import MoveHistory, db, bcrypt, User, GameSession, Match
from routes import init_routes
import chess
from chess_detection import ChessDetectionService
import cv2
import time

app = Flask(__name__)
app.config.from_object(Config)

db.init_app(app)
bcrypt.init_app(app)
socketio = SocketIO(app, cors_allowed_origins="*")

login_manager = LoginManager(app)
login_manager.login_view = 'login'

with app.app_context():
    # Drop semua tabel dan buat ulang
    # db.drop_all()
    db.create_all()
    
    # Tambahkan user admin jika belum ada
    if not User.query.filter_by(username='admin').first():
        admin = User(username='admin', role='admin')
        admin.set_password('admin123')
        db.session.add(admin)
        db.session.commit()
        print("Admin user created successfully")

init_routes(app, login_manager)

# Socket.IO Events
@socketio.on('join_match')
def on_join_match(data):
    match_id = data['match_id']
    join_room(f"match_{match_id}")
    emit('status', {'msg': f'Joined match {match_id}'})

@socketio.on('timer_update')
def on_timer_update(data):
    match_id = data['match_id']
    session = GameSession.query.filter_by(match_id=match_id).first()
    
    if session:
        if data['player'] == 1:
            session.player1_time = data['time']
        else:
            session.player2_time = data['time']
        
        session.current_player = data['current_player']
        session.is_active = data['is_active']
        db.session.commit()
        
        # Broadcast ke semua client di room
        socketio.emit('timer_sync', {
            'player1_time': session.player1_time,
            'player2_time': session.player2_time,
            'current_player': session.current_player,
            'is_active': session.is_active
        }, room=f"match_{match_id}")

@socketio.on('switch_player')
def on_switch_player(data):
    match_id = data['match_id']
    session = GameSession.query.filter_by(match_id=match_id).first()
    
    if session:
        session.current_player = 2 if session.current_player == 1 else 1
        db.session.commit()
        
        socketio.emit('player_switched', {
            'current_player': session.current_player
        }, room=f"match_{match_id}")

@socketio.on('admin_control')
def on_admin_control(data):
    match_id = data['match_id']
    action = data['action']
    session = GameSession.query.filter_by(match_id=match_id).first()
    match = Match.query.get(match_id)
    
    if session and match:
        if action == 'start':
            session.is_active = True
        elif action == 'pause':
            session.is_active = False
        elif action == 'reset':
            # Reset ke waktu sesuai mode yang dipilih saat match dibuat
            session.player1_time = match.initial_time
            session.player2_time = match.initial_time
            session.current_player = 1
            session.is_active = False
        
        db.session.commit()
        
        socketio.emit('timer_sync', {
            'player1_time': session.player1_time,
            'player2_time': session.player2_time,
            'current_player': session.current_player,
            'is_active': session.is_active,
            'time_control': match.time_control,  # Tambahkan info time control
            'initial_time': match.initial_time,   # Tambahkan info waktu awal
            'increment': match.increment          # Tambahkan info increment
        }, room=f"match_{match_id}")
detection_service = ChessDetectionService()

@socketio.on('update_detection_config')
def on_update_detection_config(data):
    try:
        camera_index = data.get('camera_index')
        mode = data.get('mode')
        show_bbox = data.get('show_bbox')
        
        # Update detection settings
        detection_service.update_detection_settings(
            camera_index=int(camera_index) if camera_index is not None else None,
            mode=mode,
            show_bbox=show_bbox
        )
        
        # Broadcast update to all clients
        socketio.emit('detection_config_updated', {
            'camera_index': detection_service.camera_index,
            'mode': detection_service.detection_mode,
            'show_bbox': detection_service.show_bbox
        })
        
    except Exception as e:
        emit('error', {'message': f'Error updating detection config: {str(e)}'})

@app.route('/api/detect_fen')
def detect_fen():
    """Endpoint untuk mendapatkan FEN dari deteksi real-time - menggunakan metode yang sama dengan mode Grid"""
    try:
        # Cek apakah detection_service sudah diinisialisasi
        if not hasattr(detection_service, 'model') or detection_service.model is None:
            return jsonify({
                'error': 'Chess detection model not loaded',
                'success': False
            }), 500

        # Ambil frame dari kamera dengan warming-up
        cap = None
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return jsonify({
                    'error': 'Could not open camera',
                    'success': False
                }), 500
            
            # Set resolusi untuk mempercepat capture (opsional)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            print("Camera opened, warming up...")
            
            # Warming up kamera - capture beberapa frame dan buang
            # Ini untuk melewati logo smart connect
            for i in range(10):  # Capture 10 frame untuk warming up
                ret, frame = cap.read()
                if ret:
                    print(f"Warming up frame {i+1}/10")
                time.sleep(0.1)  # Delay 100ms antar frame
            
            # Delay tambahan untuk memastikan kamera siap
            print("Additional delay...")
            time.sleep(1.0)  # Delay 1 detik tambahan
            
            # Sekarang capture frame yang sebenarnya
            print("Capturing actual frame...")
            ret, frame = cap.read()
            
        except Exception as e:
            return jsonify({
                'error': f'Camera error: {str(e)}',
                'success': False
            }), 500
        finally:
            if cap is not None:
                cap.release()
        
        if not ret or frame is None:
            return jsonify({
                'error': 'Could not capture frame from camera',
                'success': False
            }), 500
        
        print(f"Frame captured successfully, size: {frame.shape}")
        
        # Buat folder untuk menyimpan hasil deteksi jika belum ada
        detection_folder = 'app/static/detection_results'
        os.makedirs(detection_folder, exist_ok=True)
        
        # Generate timestamp untuk nama file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # microseconds to milliseconds
        
        # Simpan frame original
        original_filename = f"original_{timestamp}.jpg"
        original_path = os.path.join(detection_folder, original_filename)
        cv2.imwrite(original_path, frame)
        print(f"Original frame saved: {original_path}")
        
        # Gunakan method detect_pieces yang sama seperti di realtime detection
        try:
            print("Starting piece detection...")
            processed_frame, piece_results, board_corners, grid_coords, fen_code = detection_service.detect_pieces(
                frame, 
                mode='raw',              # atau 'clahe' sesuai kebutuhan
                show_bbox=True,          # aktifkan bbox untuk visualisasi hasil
                show_board_grid=True,    # aktifkan grid untuk visualisasi hasil
                use_flattened=True       # gunakan flattened mode seperti Grid
            )
            print(f"Detection completed. FEN: {fen_code}")
        except Exception as e:
            print(f"Detection error: {str(e)}")
            return jsonify({
                'error': f'Detection error: {str(e)}',
                'success': False,
                'saved_images': {
                    'original': f"/static/detection_results/{original_filename}"
                }
            }), 500
        
        # Simpan processed frame (hasil deteksi dengan bbox dan grid)
        processed_filename = f"processed_{timestamp}.jpg"
        processed_path = os.path.join(detection_folder, processed_filename)
        if processed_frame is not None:
            cv2.imwrite(processed_path, processed_frame)
            print(f"Processed frame saved: {processed_path}")
        
        # Simpan flattened board jika ada
        flattened_filename = None
        if board_corners is not None:
            try:
                # Ambil flattened board menggunakan homography
                flattened_board = detection_service.apply_homography(frame, board_corners)
                if flattened_board is not None:
                    flattened_filename = f"flattened_{timestamp}.jpg"
                    flattened_path = os.path.join(detection_folder, flattened_filename)
                    cv2.imwrite(flattened_path, flattened_board)
                    print(f"Flattened board saved: {flattened_path}")
            except Exception as e:
                print(f"Error saving flattened board: {e}")
        
        # Hitung informasi tambahan untuk debugging
        pieces_count = len(piece_results.boxes) if piece_results and piece_results.boxes else 0
        board_detected = board_corners is not None and grid_coords is not None
        grid_squares = len(grid_coords) if grid_coords else 0
        
        print(f"Detection summary: pieces={pieces_count}, board_detected={board_detected}, grid_squares={grid_squares}")
        
        # Prepare saved images info
        saved_images = {
            'original': f"/static/detection_results/{original_filename}",
            'processed': f"/static/detection_results/{processed_filename}" if processed_frame is not None else None,
            'flattened': f"/static/detection_results/{flattened_filename}" if flattened_filename else None
        }
        
        if fen_code:
            return jsonify({
                'fen': fen_code,
                'success': True,
                'pieces_detected': pieces_count,
                'board_detected': board_detected,
                'grid_squares': grid_squares,
                'saved_images': saved_images,
                'timestamp': timestamp,
                'debug_info': {
                    'has_piece_results': piece_results is not None,
                    'has_board_corners': board_corners is not None,
                    'has_grid_coords': grid_coords is not None,
                    'fen_length': len(fen_code) if fen_code else 0,
                    'board_corners': board_corners.tolist() if board_corners is not None else None
                }
            })
        else:
            return jsonify({
                'error': 'Could not generate FEN - board or pieces not detected properly',
                'success': False,
                'saved_images': saved_images,
                'timestamp': timestamp,
                'debug_info': {
                    'pieces_detected': pieces_count,
                    'board_detected': board_detected,
                    'grid_squares': grid_squares,
                    'has_piece_results': piece_results is not None,
                    'has_board_corners': board_corners is not None,
                    'has_grid_coords': grid_coords is not None
                }
            }), 400
            
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify({
            'error': f'Unexpected error: {str(e)}',
            'success': False
        }), 500

@socketio.on('start_timer_with_fen')
def on_start_timer_with_fen(data):
    """Handler untuk start timer dengan FEN awal"""
    print(f"=== START TIMER WITH FEN CALLED ===")
    print(f"Data received: {data}")
    
    match_id = data['match_id']
    initial_fen = data.get('initial_fen')
    print(f"Initial FEN: {initial_fen}")
    
    print(f"Match ID: {match_id}, Initial FEN: {initial_fen}")
    
    session = GameSession.query.filter_by(match_id=match_id).first()
    print(f"Session found: {session}")
    
    if session:
        session.is_active = True
        session.last_move_time = datetime.now()
        db.session.commit()
        print(f"Session updated, current_player: {session.current_player}")
        
        # Simpan FEN awal sebagai move pertama dengan fen_after = None
        if initial_fen:
            # Hapus record lama jika ada (untuk restart)
            deleted_count = MoveHistory.query.filter_by(match_id=match_id).delete()
            print(f"Deleted {deleted_count} old move records")
            db.session.commit()
            
            move = MoveHistory(
                match_id=match_id,
                move_number=1,  # Mulai dari move 1
                player=session.current_player,  # Player 1 (putih) yang mulai
                fen_before=initial_fen,
                fen_after=None,  # Belum ada move
                uci_move=None
            )
            db.session.add(move)
            
            try:
                db.session.commit()
                print(f"✅ Initial move record created successfully! ID: {move.id}")
                
                # Verifikasi record tersimpan
                saved_move = MoveHistory.query.filter_by(match_id=match_id, move_number=1).first()
                if saved_move:
                    print(f"✅ Verification: Move record exists with ID {saved_move.id}")
                else:
                    print("❌ Verification failed: Move record not found after commit")
                    
            except Exception as e:
                print(f"❌ Error saving initial move: {e}")
                db.session.rollback()
                return
            
            print(f"Initial FEN saved for match {match_id}: {initial_fen}, player: {session.current_player}")
        else:
            print("❌ No initial FEN provided")
        
        socketio.emit('timer_started', {
            'match_id': match_id,
            'initial_fen': initial_fen,
            'current_player': session.current_player
        }, room=f"match_{match_id}")
        print(f"✅ timer_started event emitted")
    else:
        print(f"❌ No session found for match_id: {match_id}")

@socketio.on('move_history')
def on_move_history(data):
    """Handler untuk update move history saat switch player"""
    print(f"=== MOVE HISTORY CALLED ===")
    print(f"Data received: {data}")
    
    match_id = data['match_id']
    player = data['player']  # Player yang baru saja selesai movenya
    fen_after = data.get('fen_after')
    
    print(f"Match ID: {match_id}, Player: {player}, FEN After: {fen_after}")
    
    # Validasi FEN
    if not fen_after:
        print("❌ Error: Missing FEN after move")
        return
    
    try:
        # Validasi FEN format
        test_board = chess.Board(fen_after)
        print(f"✅ FEN validation passed: {len(fen_after)} characters")
    except Exception as e:
        print(f"❌ Invalid FEN format: {e}")
        return
    
    try:
        # Cari record terakhir yang belum selesai (fen_after masih None)
        incomplete_move = MoveHistory.query.filter_by(
            match_id=match_id, 
            fen_after=None
        ).order_by(MoveHistory.move_number.desc()).first()
        
        print(f"🔍 Incomplete move found: {incomplete_move}")
        if incomplete_move:
            print(f"📝 Incomplete move details: ID={incomplete_move.id}, move_number={incomplete_move.move_number}, player={incomplete_move.player}")
            print(f"🎯 Player dari data: {player}, Player di record: {incomplete_move.player}")
        
        if incomplete_move:
            # PENTING: Pastikan player assignment benar
            # Record di database menyimpan player yang akan move next
            # Data dari frontend adalah player yang baru saja selesai move
            
            print(f"📊 Before update:")
            print(f"   fen_before: {incomplete_move.fen_before}")
            print(f"   fen_after: {incomplete_move.fen_after}")
            print(f"   player: {incomplete_move.player}")
            
            # Update record yang belum selesai dengan fen_after
            incomplete_move.fen_after = fen_after
            
            # Kalkulasi UCI move jika ada fen_before
            if incomplete_move.fen_before:
                print(f"🔄 Calculating UCI move...")
                print(f"   From: {incomplete_move.fen_before}")
                print(f"   To: {fen_after}")
                
                uci_move = get_uci_move(incomplete_move.fen_before, fen_after)
                incomplete_move.uci_move = uci_move
                
                print(f"🎯 UCI move calculated: '{uci_move}'")
                
                if uci_move:
                    print(f"✅ Move {incomplete_move.move_number} completed: Player {incomplete_move.player} - {uci_move}")
                else:
                    print(f"❌ UCI move calculation returned None/empty")
                    print(f"   Debugging FEN comparison:")
                    
                    # Debug FEN lebih detail
                    try:
                        board_before = chess.Board(incomplete_move.fen_before)
                        board_after = chess.Board(fen_after)
                        
                        print(f"   Board before turn: {board_before.turn}")
                        print(f"   Board after turn: {board_after.turn}")
                        print(f"   Legal moves count: {len(list(board_before.legal_moves))}")
                        
                        # Coba manual check beberapa moves
                        for i, move in enumerate(board_before.legal_moves):
                            if i >= 5:  # Hanya cek 5 move pertama
                                break
                            board_test = board_before.copy()
                            board_test.push(move)
                            print(f"   Testing move {move}: {board_test.fen()[:20]}...")
                            
                    except Exception as debug_e:
                        print(f"   Debug error: {debug_e}")
                        
            else:
                print("❌ No fen_before found for UCI calculation")
            
            print(f"📊 After update:")
            print(f"   fen_before: {incomplete_move.fen_before}")
            print(f"   fen_after: {incomplete_move.fen_after}")
            print(f"   uci_move: {incomplete_move.uci_move}")
            print(f"   player: {incomplete_move.player}")
            
            try:
                db.session.commit()
                print(f"✅ Move {incomplete_move.move_number} updated successfully")
            except Exception as e:
                print(f"❌ Error updating move: {e}")
                db.session.rollback()
                return
            
            # Broadcast completed move
            socketio.emit('move_completed', {
                'move_number': incomplete_move.move_number,
                'player': incomplete_move.player,  # Use player from record, not from data
                'uci_move': incomplete_move.uci_move,
                'fen_before': incomplete_move.fen_before,
                'fen_after': fen_after
            }, room=f"match_{match_id}")
            print(f"✅ move_completed event emitted")
            
            # Buat record baru untuk move berikutnya
            next_player = 2 if incomplete_move.player == 1 else 1  # Use player from record
            next_move_number = incomplete_move.move_number + 1
            
            print(f"🆕 Creating next move record:")
            print(f"   move_number: {next_move_number}")
            print(f"   player: {next_player}")
            print(f"   fen_before: {fen_after}")
            
            new_move = MoveHistory(
                match_id=match_id,
                move_number=next_move_number,
                player=next_player,
                fen_before=fen_after,
                fen_after=None,
                uci_move=None
            )
            db.session.add(new_move)
            
            try:
                db.session.commit()
                print(f"✅ New move record created: Move {next_move_number}, Player {next_player}, ID: {new_move.id}")
            except Exception as e:
                print(f"❌ Error creating new move record: {e}")
                db.session.rollback()
                return
            
        else:
            print("❌ Error: No incomplete move found to update")
            
            # Debug: tampilkan semua move history untuk match ini
            all_moves = MoveHistory.query.filter_by(match_id=match_id).order_by(MoveHistory.move_number).all()
            print(f"📋 All moves for match {match_id} ({len(all_moves)} total):")
            for move in all_moves:
                print(f"  Move {move.move_number}: Player {move.player}, "
                      f"fen_before: {'Yes' if move.fen_before else 'None'}, "
                      f"fen_after: {'Yes' if move.fen_after else 'None'}, "
                      f"uci_move: {move.uci_move or 'None'}")
        
    except Exception as e:
        print(f"❌ Error updating move history: {e}")
        import traceback
        traceback.print_exc()
@socketio.on('move_history')
def on_move_history(data):
    """Handler untuk update move history saat switch player"""
    print(f"=== MOVE HISTORY CALLED ===")
    print(f"Data received: {data}")
    
    match_id = data['match_id']
    player = data['player']  # Player yang baru saja selesai movenya
    fen_after = data.get('fen_after')
    
    print(f"Match ID: {match_id}, Player: {player}, FEN After: {fen_after}")
    
    # Validasi FEN
    if not fen_after:
        print("❌ Error: Missing FEN after move")
        return
    
    try:
        # Validasi FEN format
        test_board = chess.Board(fen_after)
        print(f"✅ FEN validation passed: {len(fen_after)} characters")
    except Exception as e:
        print(f"❌ Invalid FEN format: {e}")
        return
    
    try:
        # Cari record terakhir yang belum selesai (fen_after masih None)
        incomplete_move = MoveHistory.query.filter_by(
            match_id=match_id, 
            fen_after=None
        ).order_by(MoveHistory.move_number.desc()).first()
        
        print(f"🔍 Incomplete move found: {incomplete_move}")
        if incomplete_move:
            print(f"📝 Incomplete move details: ID={incomplete_move.id}, move_number={incomplete_move.move_number}, player={incomplete_move.player}")
            print(f"🎯 Player dari data: {player}, Player di record: {incomplete_move.player}")
        
        if incomplete_move:
            print(f"📊 Before update:")
            print(f"   fen_before: {incomplete_move.fen_before}")
            print(f"   fen_after: {incomplete_move.fen_after}")
            print(f"   player: {incomplete_move.player}")
            
            # AUTO-FIX FEN dengan turn yang benar sebelum simpan
            fen_before_fixed = auto_fix_fen_turn(incomplete_move.fen_before, incomplete_move.move_number)
            fen_after_fixed = auto_fix_fen_turn(fen_after, incomplete_move.move_number + 1)
            
            print(f"🔧 FEN auto-fixed:")
            print(f"   fen_before_fixed: {fen_before_fixed}")
            print(f"   fen_after_fixed: {fen_after_fixed}")
            
            # Update record dengan FEN yang sudah difix
            incomplete_move.fen_before = fen_before_fixed  # Update fen_before juga
            incomplete_move.fen_after = fen_after_fixed
            
            # Kalkulasi UCI move dengan FEN yang benar
            if fen_before_fixed:
                print(f"🔄 Calculating UCI move...")
                print(f"   From: {fen_before_fixed}")
                print(f"   To: {fen_after_fixed}")
                
                # Pass move_number untuk auto-fix di get_uci_move
                uci_move = get_uci_move(fen_before_fixed, fen_after_fixed, incomplete_move.move_number)
                incomplete_move.uci_move = uci_move
                
                print(f"🎯 UCI move calculated: '{uci_move}'")
                
                if uci_move:
                    print(f"✅ Move {incomplete_move.move_number} completed: Player {incomplete_move.player} - {uci_move}")
                else:
                    print(f"❌ UCI move calculation returned None/empty")
                        
            else:
                print("❌ No fen_before found for UCI calculation")
            
            print(f"📊 After update:")
            print(f"   fen_before: {incomplete_move.fen_before}")
            print(f"   fen_after: {incomplete_move.fen_after}")
            print(f"   uci_move: {incomplete_move.uci_move}")
            print(f"   player: {incomplete_move.player}")
            
            try:
                db.session.commit()
                print(f"✅ Move {incomplete_move.move_number} updated successfully")
            except Exception as e:
                print(f"❌ Error updating move: {e}")
                db.session.rollback()
                return
            
            # Broadcast completed move
            socketio.emit('move_completed', {
                'move_number': incomplete_move.move_number,
                'player': incomplete_move.player,
                'uci_move': incomplete_move.uci_move,
                'fen_before': incomplete_move.fen_before,
                'fen_after': incomplete_move.fen_after
            }, room=f"match_{match_id}")
            print(f"✅ move_completed event emitted")
            
            # Buat record baru untuk move berikutnya
            next_player = 2 if incomplete_move.player == 1 else 1
            next_move_number = incomplete_move.move_number + 1
            
            print(f"🆕 Creating next move record:")
            print(f"   move_number: {next_move_number}")
            print(f"   player: {next_player}")
            print(f"   fen_before: {fen_after_fixed}")
            
            new_move = MoveHistory(
                match_id=match_id,
                move_number=next_move_number,
                player=next_player,
                fen_before=fen_after_fixed,  # Gunakan FEN yang sudah difix
                fen_after=None,
                uci_move=None
            )
            db.session.add(new_move)
            
            try:
                db.session.commit()
                print(f"✅ New move record created: Move {next_move_number}, Player {next_player}, ID: {new_move.id}")
            except Exception as e:
                print(f"❌ Error creating new move record: {e}")
                db.session.rollback()
                return
            
        else:
            print("❌ Error: No incomplete move found to update")
            
            # Debug: tampilkan semua move history untuk match ini
            all_moves = MoveHistory.query.filter_by(match_id=match_id).order_by(MoveHistory.move_number).all()
            print(f"📋 All moves for match {match_id} ({len(all_moves)} total):")
            for move in all_moves:
                print(f"  Move {move.move_number}: Player {move.player}, "
                      f"fen_before: {'Yes' if move.fen_before else 'None'}, "
                      f"fen_after: {'Yes' if move.fen_after else 'None'}, "
                      f"uci_move: {move.uci_move or 'None'}")
        
    except Exception as e:
        print(f"❌ Error updating move history: {e}")
        import traceback
        traceback.print_exc()



def get_uci_move(fen_before, fen_after):
    """
    Kalkulasi UCI move dari FEN before ke FEN after
    GUNAKAN STRING FEN ORIGINAL TANPA RE-PARSING
    """
    try:
        if not fen_before or not fen_after:
            print("❌ get_uci_move: Missing FEN")
            return None
            
        print(f"🔄 get_uci_move called:")
        print(f"   fen_before (original): {fen_before}")
        print(f"   fen_after (original): {fen_after}")
        
        # GUNAKAN FEN STRING ORIGINAL - JANGAN RE-PARSE!
        pieces_before = fen_before.split(' ')[0]  # Ambil langsung dari string
        pieces_after = fen_after.split(' ')[0]    # Ambil langsung dari string
        
        print(f"   Pieces before (direct): {pieces_before}")
        print(f"   Pieces after (direct): {pieces_after}")
        
        if pieces_before == pieces_after:
            print("❌ No piece movement detected")
            return None
        
        # Parse board HANYA untuk legal moves
        board_before = chess.Board(fen_before)
        
        print(f"   Board turn: {'White' if board_before.turn else 'Black'}")
        print(f"   Total legal moves: {len(list(board_before.legal_moves))}")
        
        # Cari move yang valid
        legal_moves = list(board_before.legal_moves)
        
        for i, move in enumerate(legal_moves):
            # Test move
            board_test = board_before.copy()
            board_test.push(move)
            
            # Ambil pieces LANGSUNG dari FEN hasil test
            test_pieces = board_test.fen().split(' ')[0]
            
            # Bandingkan dengan pieces_after (dari string original)
            if test_pieces == pieces_after:
                print(f"✅ Found matching move: {move.uci()} (checked {i+1}/{len(legal_moves)})")
                return move.uci()
                
            # Debug untuk move pertama beberapa
            if i < 5:
                print(f"   Move {move.uci()}: {test_pieces[:15]}... != {pieces_after[:15]}...")
        
        print(f"❌ No matching move found after checking {len(legal_moves)} moves")
        
        # DEBUGGING TAMBAHAN: Cek apakah ada subtle differences
        print(f"\n🔍 DETAILED COMPARISON:")
        print(f"   Target pieces: '{pieces_after}'")
        print(f"   Length: {len(pieces_after)}")
        
        # Test beberapa moves lagi dengan detail
        for i, move in enumerate(legal_moves[:10]):
            board_test = board_before.copy()
            board_test.push(move)
            test_pieces = board_test.fen().split(' ')[0]
            
            # Hitung character differences
            if len(test_pieces) == len(pieces_after):
                diff_chars = [j for j, (a, b) in enumerate(zip(test_pieces, pieces_after)) if a != b]
                if len(diff_chars) <= 3:  # Jika hanya sedikit perbedaan
                    print(f"   Move {move.uci()}: {len(diff_chars)} differences at positions {diff_chars}")
                    print(f"     Test:   '{test_pieces}'")
                    print(f"     Target: '{pieces_after}'")
        
        return None
        
    except Exception as e:
        print(f"❌ UCI move generation error: {e}")
        import traceback
        traceback.print_exc()
        return None
    
def auto_fix_fen_turn(fen_string, move_number):
    """
    Auto fix FEN turn berdasarkan move number
    Move ganjil (1,3,5...) = White to move
    Move genap (2,4,6...) = Black to move
    """
    try:
        parts = fen_string.split(' ')
        if len(parts) >= 6:
            # Tentukan turn berdasarkan move number
            expected_turn = 'w' if move_number % 2 == 1 else 'b'
            parts[1] = expected_turn
            
            # Update fullmove number juga
            fullmove_number = (move_number + 1) // 2
            parts[5] = str(fullmove_number)
            
            return ' '.join(parts)
        return fen_string
    except:
        return fen_string
        
def get_uci_move(fen_before, fen_after, move_number=None):
    """
    Kalkulasi UCI move dengan auto-fix turn
    """
    try:
        if not fen_before or not fen_after:
            print("❌ get_uci_move: Missing FEN")
            return None
            
        print(f"🔄 get_uci_move called:")
        print(f"   fen_before (original): {fen_before}")
        print(f"   fen_after (original): {fen_after}")
        print(f"   move_number: {move_number}")
        
        # AUTO-FIX TURN jika move_number ada
        if move_number:
            # Move number menentukan siapa yang akan move
            # Move 1 = White to move, Move 2 = Black to move, dst.
            fen_before_fixed = auto_fix_fen_turn(fen_before, move_number)
            
            # Setelah move, turn berganti
            next_move_number = move_number + 1
            fen_after_fixed = auto_fix_fen_turn(fen_after, next_move_number)
            
            print(f"🔧 Turn auto-fixed:")
            print(f"   fen_before_fixed: {fen_before_fixed}")
            print(f"   fen_after_fixed: {fen_after_fixed}")
        else:
            fen_before_fixed = fen_before
            fen_after_fixed = fen_after
        
        # Ambil pieces position
        pieces_before = fen_before_fixed.split(' ')[0]
        pieces_after = fen_after_fixed.split(' ')[0]
        
        print(f"   Pieces before: {pieces_before}")
        print(f"   Pieces after: {pieces_after}")
        
        if pieces_before == pieces_after:
            print("❌ No piece movement detected")
            return None
        
        # Parse board dengan turn yang benar
        board_before = chess.Board(fen_before_fixed)
        
        print(f"   Board turn: {'White' if board_before.turn else 'Black'}")
        print(f"   Total legal moves: {len(list(board_before.legal_moves))}")
        
        # Cari move yang valid
        legal_moves = list(board_before.legal_moves)
        
        for i, move in enumerate(legal_moves):
            board_test = board_before.copy()
            board_test.push(move)
            
            test_pieces = board_test.fen().split(' ')[0]
            
            if test_pieces == pieces_after:
                print(f"✅ Found matching move: {move.uci()} (checked {i+1}/{len(legal_moves)})")
                return move.uci()
                
            if i < 5:
                print(f"   Move {move.uci()}: {test_pieces[:15]}... != {pieces_after[:15]}...")
        
        print(f"❌ No matching move found after checking {len(legal_moves)} moves")
        return None
        
    except Exception as e:
        print(f"❌ UCI move generation error: {e}")
        import traceback
        traceback.print_exc()
        return None

@socketio.on('start_opencv_detection')
def on_start_opencv_detection(data):
    try:
        camera_index = int(data.get('camera_index', 0))
        mode = data.get('mode', 'raw')
        show_bbox = data.get('show_bbox', True)
        
        success = detection_service.start_opencv_detection(camera_index, mode, show_bbox)
        
        if success:
            socketio.emit('opencv_detection_started', {
                'camera_index': camera_index,
                'mode': mode,
                'show_bbox': show_bbox
            })
        else:
            emit('error', {'message': 'Failed to start OpenCV detection'})
            
    except Exception as e:
        emit('error', {'message': f'Error starting detection: {str(e)}'})

@socketio.on('stop_opencv_detection')
def on_stop_opencv_detection():
    try:
        detection_service.stop_opencv_detection()
        socketio.emit('opencv_detection_stopped')
    except Exception as e:
        emit('error', {'message': f'Error stopping detection: {str(e)}'})

if __name__ == '__main__':
    socketio.run(app, debug=True)