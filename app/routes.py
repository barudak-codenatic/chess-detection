from flask import render_template, redirect, url_for, request, flash, jsonify
from flask_login import login_user, logout_user, current_user, login_required
from models import db, User, Match, GameSession
from chess_detection import ChessDetectionService
import base64
import cv2
import numpy as np

def init_routes(app, login_manager):

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    @app.route('/')
    def index():
        return redirect(url_for('login'))

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']
            user = User.query.filter_by(username=username).first()
            if user and user.check_password(password):
                login_user(user)
                if user.role == 'admin':
                    return redirect(url_for('admin_dashboard'))
                else:
                    return redirect(url_for('player_dashboard'))
            else:
                flash('Username atau password salah', 'danger')
        return render_template('login.html')

    @app.route('/logout')
    def logout():
        logout_user()
        return redirect(url_for('login'))

    @app.route('/admin')
    @login_required
    def admin_dashboard():
        if current_user.role != 'admin':
            return redirect(url_for('player_dashboard'))
        players = User.query.filter_by(role='player').all()
        matches = Match.query.all()
        return render_template('dashboard_admin.html', players=players, matches=matches)

    @app.route('/admin/create_player', methods=['POST'])
    @login_required
    def create_player():
        if current_user.role != 'admin':
            return redirect(url_for('player_dashboard'))
        username = request.form['username']
        password = request.form['password']
        
        if User.query.filter_by(username=username).first():
            flash('Username sudah ada', 'danger')
            return redirect(url_for('admin_dashboard'))
        
        new_player = User(username=username, role='player')
        new_player.set_password(password)
        db.session.add(new_player)
        db.session.commit()
        flash('Player berhasil ditambahkan', 'success')
        return redirect(url_for('admin_dashboard'))

    @app.route('/admin/create_match', methods=['POST'])
    @login_required
    def create_match():
        if current_user.role != 'admin':
            return redirect(url_for('player_dashboard'))
        player1_id = request.form['player1_id']
        player2_id = request.form['player2_id']
        
        if player1_id == player2_id:
            flash('Player 1 dan Player 2 harus berbeda', 'danger')
            return redirect(url_for('admin_dashboard'))
        
        new_match = Match(player1_id=player1_id, player2_id=player2_id)
        db.session.add(new_match)
        db.session.commit()
        
        # Buat game session
        game_session = GameSession(match_id=new_match.id, current_player=1)
        db.session.add(game_session)
        db.session.commit()
        
        flash('Pertandingan berhasil dibuat', 'success')
        return redirect(url_for('admin_dashboard'))

    @app.route('/player')
    @login_required
    def player_dashboard():
        if current_user.role != 'player':
            return redirect(url_for('admin_dashboard'))
        matches = Match.query.filter(
            (Match.player1_id == current_user.id) |
            (Match.player2_id == current_user.id)
        ).all()
        return render_template('dashboard_player.html', matches=matches)

    @app.route('/match/<int:match_id>')
    @login_required
    def match_detail(match_id):
        match = Match.query.get_or_404(match_id)
        # Cek apakah user berhak akses match ini
        if current_user.role != 'admin' and current_user.id not in [match.player1_id, match.player2_id]:
            flash('Anda tidak memiliki akses ke pertandingan ini', 'danger')
            return redirect(url_for('player_dashboard'))
        
        # Get atau buat game session
        session = GameSession.query.filter_by(match_id=match_id).first()
        if not session:
            session = GameSession(match_id=match_id, current_player=1)
            db.session.add(session)
            db.session.commit()
        
        if current_user.role == 'admin':
            return render_template('match_detail_admin.html', match=match, session=session)
        else:
            return render_template('match_detail_player.html', match=match, session=session)

    @app.route('/api/session/<int:match_id>')
    @login_required
    def get_session(match_id):
        session = GameSession.query.filter_by(match_id=match_id).first()
        if session:
            return jsonify({
                'player1_time': session.player1_time,
                'player2_time': session.player2_time,
                'current_player': session.current_player,
                'is_active': session.is_active
            })
        return jsonify({'error': 'Session not found'}), 404

    # Initialize detection service
    detection_service = ChessDetectionService()

    @app.route('/api/detect', methods=['POST'])
    @login_required
    def detect_chess():
        if current_user.role != 'admin':
            return jsonify({'error': 'Unauthorized'}), 403
        
        try:
            data = request.get_json()
            if not data or 'image' not in data:
                return jsonify({'error': 'No image data provided'}), 400
                
            image_data = data.get('image')
            mode = data.get('mode', 'raw')
            show_bbox = data.get('show_bbox', True)
            
            # Decode base64 image
            try:
                image_data_clean = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data_clean)
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is None:
                    return jsonify({'error': 'Failed to decode image'}), 400
                    
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except Exception as e:
                return jsonify({'error': f'Image decoding error: {str(e)}'}), 400
            
            # Run detection
            result_image, detection_results = detection_service.detect_pieces(image, mode, show_bbox)
            detections = detection_service.get_detection_data(detection_results)
            
            # Encode result image back to base64
            try:
                result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
                _, buffer = cv2.imencode('.jpg', result_image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                result_base64 = base64.b64encode(buffer).decode('utf-8')
            except Exception as e:
                return jsonify({'error': f'Image encoding error: {str(e)}'}), 500
            
            return jsonify({
                'image': f"data:image/jpeg;base64,{result_base64}",
                'detections': detections,
                'piece_count': len(detections),
                'mode': mode,
                'show_bbox': show_bbox
            })
            
        except Exception as e:
            return jsonify({'error': f'Detection service error: {str(e)}'}), 500