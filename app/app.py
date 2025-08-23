from flask import Flask
from flask_login import LoginManager
from flask_socketio import SocketIO, emit, join_room, leave_room
from config import Config
from models import db, bcrypt, User, GameSession, Match
from routes import init_routes
from chess_detection import ChessDetectionService

app = Flask(__name__)
app.config.from_object(Config)

db.init_app(app)
bcrypt.init_app(app)
socketio = SocketIO(app, cors_allowed_origins="*")

login_manager = LoginManager(app)
login_manager.login_view = 'login'

with app.app_context():
    # Drop semua tabel dan buat ulang
    db.drop_all()
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