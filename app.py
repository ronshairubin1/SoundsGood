import os
import io
import json
import logging
import threading
from datetime import datetime

from flask import (
    Flask, render_template, request, session, redirect,
    url_for, flash, jsonify
)
from flask_cors import CORS

from config import Config

# Import our blueprints
from routes.ml_routes import ml_bp
from src.api.dictionary_api import dictionary_bp
from src.api.user_api import user_bp

# Import services
from src.services.dictionary_service import DictionaryService
from src.services.user_service import UserService

# --------------------------------------------------------------------
# Set up Flask
# --------------------------------------------------------------------
app = Flask(
    __name__,
    static_url_path='/static',
    static_folder=os.path.join(Config.BASE_DIR, 'static'),
    template_folder=os.path.join(Config.BASE_DIR, 'templates')
)
app.secret_key = Config.SECRET_KEY
CORS(app, supports_credentials=True)

# Register blueprints
app.register_blueprint(ml_bp, url_prefix='/ml')
app.register_blueprint(dictionary_bp)
app.register_blueprint(user_bp)

# Initialize services
dictionary_service = DictionaryService()
user_service = UserService()

# --------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------
logging.basicConfig(level=logging.DEBUG)
app.logger.debug("Starting Sound Classifier app...")
app.logger.debug(f"Template folder: {app.template_folder}")
app.logger.debug(f"Static folder: {app.static_folder}")

# --------------------------------------------------------------------
# Initialize directories
# --------------------------------------------------------------------
Config.init_directories()

# --------------------------------------------------------------------
# Basic Routes: index, login, logout, register
# --------------------------------------------------------------------
@app.route('/')
def index():
    # If not logged in, ask for login
    if 'username' not in session:
        return render_template('login.html')
    # If logged in, go to dashboard
    return render_template('dashboard.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            flash('Please provide both username and password', 'danger')
            return render_template('login.html')
        
        # Use the user service to authenticate
        result = user_service.login(username, password)
        
        if result['success']:
            # Set session data
            session['username'] = username
            session['user_id'] = result['user']['id']
            session['is_admin'] = result['user']['is_admin']
            flash(f'Welcome back, {username}!', 'success')
            return redirect(url_for('index'))
        else:
            flash(result['error'], 'danger')
            return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        email = request.form.get('email')
        
        if not username or not password:
            flash('Please provide both username and password', 'danger')
            return render_template('register.html')
        
        # Use the user service to register
        result = user_service.register(username, password, email)
        
        if result['success']:
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        else:
            flash(result['error'], 'danger')
            return render_template('register.html')

# --------------------------------------------------------------------
# Dictionary Management Routes
# --------------------------------------------------------------------
@app.route('/dictionaries/manage')
def manage_dictionaries():
    """Render the dictionary management page."""
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Get dictionaries
    dictionaries = dictionary_service.get_dictionaries(session.get('user_id'))
    
    return render_template('manage_dictionaries.html', dictionaries=dictionaries)

@app.route('/dictionaries/<dict_name>/view')
def view_dictionary(dict_name):
    """Render the dictionary view page."""
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Get dictionary
    dictionary = dictionary_service.get_dictionary(dict_name)
    
    if not dictionary:
        flash(f'Dictionary "{dict_name}" not found', 'danger')
        return redirect(url_for('manage_dictionaries'))
    
    return render_template('dictionary_view.html', dictionary=dictionary)

@app.route('/dictionaries/<dict_name>/edit')
def edit_dictionary(dict_name):
    """Render the dictionary edit page."""
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Get dictionary
    dictionary = dictionary_service.get_dictionary(dict_name)
    
    if not dictionary:
        flash(f'Dictionary "{dict_name}" not found', 'danger')
        return redirect(url_for('manage_dictionaries'))
    
    return render_template('dictionary_edit.html', dictionary=dictionary)

# --------------------------------------------------------------------
# Training and Prediction Routes
# --------------------------------------------------------------------
@app.route('/training')
def training():
    """Render the training page."""
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Get dictionaries for selecting which to train on
    dictionaries = dictionary_service.get_dictionaries(session.get('user_id'))
    
    # Get selected dictionary if provided in query string
    selected_dict = request.args.get('dictionary')
    
    return render_template('training.html', 
                          dictionaries=dictionaries,
                          selected_dict=selected_dict)

@app.route('/predict')
def predict():
    """Render the prediction page."""
    if 'username' not in session:
        return redirect(url_for('login'))
    
    return render_template('predict.html')

@app.route('/analytics')
def analytics():
    """Render the analytics page."""
    if 'username' not in session:
        return redirect(url_for('login'))
    
    return render_template('analytics.html')

@app.route('/settings')
def settings():
    """Render the settings page."""
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Get user info
    user_info = user_service.get_user(session['username'])
    
    return render_template('settings.html', user=user_info)

# --------------------------------------------------------------------
# Before request: check login
# --------------------------------------------------------------------
@app.before_request
def check_login():
    """
    This is not strictly necessary since we're checking in each route,
    but it's a good precaution for any routes we might forget to protect.
    """
    # Skip login check for static assets, login/register routes, and API routes
    if (request.endpoint in ['login', 'register', 'static', 'logout'] or
        request.path.startswith('/static/') or
        request.path.startswith('/api/')):
        return
        
    # Require login for all other routes
    if 'username' not in session:
        if request.path != '/':
            flash('Please login to access this page', 'warning')
        return redirect(url_for('login'))

# --------------------------------------------------------------------
# Error handlers
# --------------------------------------------------------------------
@app.errorhandler(403)
def forbidden_error(e):
    flash('You do not have permission to access this resource', 'danger')
    return redirect(url_for('index'))

@app.errorhandler(401)
def unauthorized_error(e):
    flash('Please login to access this resource', 'warning')
    return redirect(url_for('login'))

@app.errorhandler(404)
def not_found_error(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    app.logger.error(f"Server error: {e}")
    return render_template('500.html', error=str(e)), 500

# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=Config.DEBUG, host='0.0.0.0', port=5000) 