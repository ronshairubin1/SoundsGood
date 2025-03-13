from flask import Flask
from backend.config import Config

def create_app(config_class=Config):
    app = Flask(__name__, static_folder='static')
    app.config.from_object(config_class)
    
    # Import blueprints
    from backend.src.routes.main_routes import main_blueprint
    from backend.src.routes.ml_routes import ml_bp
    from backend.src.routes.recording_routes import recording_bp
    from backend.src.routes.feature_routes import feature_routes
    
    # Register blueprints
    app.register_blueprint(main_blueprint)
    app.register_blueprint(ml_bp, url_prefix='/ml')
    app.register_blueprint(recording_bp, url_prefix='/recording')
    app.register_blueprint(feature_routes)
    
    # ... keep rest of the method ...

    return app 