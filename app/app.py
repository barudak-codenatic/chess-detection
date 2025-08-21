from flask import Flask
from flask_login import LoginManager
from flask_socketio import SocketIO, emit, join_room, leave_room
from config import Config
from models import db, bcrypt, User, GameSession, Match
from routes import init_routes

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
    
    if session:
        if action == 'start':
            session.is_active = True
        elif action == 'pause':
            session.is_active = False
        elif action == 'reset':
            session.player1_time = 600
            session.player2_time = 600
            session.current_player = 1
            session.is_active = False
        
        db.session.commit()
        
        socketio.emit('timer_sync', {
            'player1_time': session.player1_time,
            'player2_time': session.player2_time,
            'current_player': session.current_player,
            'is_active': session.is_active
        }, room=f"match_{match_id}")

if __name__ == '__main__':
    socketio.run(app, debug=True)