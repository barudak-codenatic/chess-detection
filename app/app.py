from flask import Flask
from flask_login import LoginManager
from config import Config
from models import db, bcrypt, User
from routes import init_routes

app = Flask(__name__)
app.config.from_object(Config)

db.init_app(app)
bcrypt.init_app(app)

login_manager = LoginManager(app)
login_manager.login_view = 'login'

with app.app_context():
    db.create_all()
    # Tambahkan user admin jika belum ada
    if not User.query.filter_by(username='admin').first():
        admin = User(username='admin', role='admin')
        admin.set_password('admin123')
        db.session.add(admin)
        db.session.commit()

init_routes(app, login_manager)

if __name__ == '__main__':
    app.run(debug=True)
