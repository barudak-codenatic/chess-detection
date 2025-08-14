from flask import render_template, redirect, url_for, request, flash
from flask_login import login_user, logout_user, current_user, login_required
from models import db, User, Match

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
        
        return render_template('match_detail.html', match=match)
