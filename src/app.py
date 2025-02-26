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

from config import Config  # Use the real config.py

# Import our new ML blueprint
from routes.ml_routes import ml_bp
from src.api.dictionary_api import dictionary_bp

# --------------------------------------------------------------------
# Set up Flask
# --------------------------------------------------------------------
app = Flask(
    __name__,
    static_url_path='/static',
    static_folder=os.path.join(Config.CURRENT_DIR, 'static'),
    template_folder=os.path.join(Config.CURRENT_DIR, 'templates')
)
app.secret_key = 'your-secret-key'
CORS(app, supports_credentials=True)

# Register the ML blueprint at /ml
app.register_blueprint(ml_bp, url_prefix='/ml')

# Register the dictionary blueprint
app.register_blueprint(dictionary_bp)

# --------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------
logging.basicConfig(level=logging.DEBUG)
app.logger.debug("Starting Sound Classifier app.py (no ML duplication)...")
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
    # If logged in, go to record page by default
    return render_template('record.html', sounds=Config.get_dictionary()['sounds'])

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')

    # Distinguish admin vs user
    if request.form.get('type') == 'admin':
        if request.form.get('password') == 'Michal':
            session['username'] = 'admin'
            session['is_admin'] = True
        else:
            flash('Invalid admin password')
            return render_template('login.html')
    else:
        username = request.form.get('username')
        if username:
            session['username'] = username
            session['is_admin'] = False
        else:
            flash('Username required')
            return render_template('login.html')
    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Simple registration route (demo only)."""
    if request.method == 'POST':
        # In a real app, you'd store the new user in a database
        username = request.form.get('username')
        if not username:
            flash("Please provide a username")
            return render_template('register.html')
        # Just log them in
        session['username'] = username
        session['is_admin'] = False
        flash(f"User '{username}' registered and logged in")
        return redirect(url_for('index'))
    return render_template('register.html')

# Basic test route
# --------------------------------------------------------------------
@app.route('/test')
def test():
    return "Server is working!"
# --------------------------------------------------------------------
# Before request: check login
# --------------------------------------------------------------------
@app.before_request
def check_login():
    if request.endpoint in ['login', 'register', 'static', 'test']:
        return
    if 'ml.' in str(request.endpoint):
        # ML routes can also require login
        pass
    if 'username' not in session:
        return redirect(url_for('login'))

# --------------------------------------------------------------------
# Error handlers
# --------------------------------------------------------------------
@app.errorhandler(403)
def forbidden_error(e):
    return redirect(url_for('login'))

@app.errorhandler(401)
def unauthorized_error(e):
    return redirect(url_for('login'))

@app.errorhandler(404)
def not_found_error(e):
    return render_template('404.html'), 404

@app.route('/dictionaries/manage')
def manage_dictionaries():
    """Render the dictionary management page."""
    # In a real app, we'd fetch dictionaries from the service
    return render_template('manage_dictionaries.html', dictionaries=[])

@app.route('/dictionaries/<dict_id>/view')
def view_dictionary(dict_id):
    """Render the dictionary view page."""
    # In a real app, we'd fetch the dictionary from the service
    return render_template('dictionary_view.html', dictionary={})

@app.route('/dictionaries/<dict_id>/edit')
def edit_dictionary(dict_id):
    """Render the dictionary edit page."""
    # In a real app, we'd fetch the dictionary from the service
    return render_template('dictionary_edit.html', dictionary={})
