from flask import Flask
from src.config import Config
from src.extensions import db

def create_app(config_class=Config):
    app = Flask(__name__, static_folder='static')
    app.config.from_object(config_class)
    db.init_app(app)

    # Import blueprints
    from src.routes.main_routes import main_blueprint
    from src.routes.ml_routes import ml_bp
    from src.routes.recording_routes import recording_bp
    
    # Register blueprints
    app.register_blueprint(main_blueprint)
    app.register_blueprint(ml_bp, url_prefix='/ml')
    app.register_blueprint(recording_bp, url_prefix='/recording')
    
    # ... keep rest of the method ...

    return app 