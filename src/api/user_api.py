import logging
from flask import request, jsonify, Blueprint, session

from src.services.user_service import UserService

# Create blueprint
user_bp = Blueprint('user', __name__, url_prefix='/api/user')
user_service = UserService()

# Initialize admin user for demo purposes
@user_bp.before_app_first_request
def init_admin():
    user_service.initialize_admin('admin', 'admin123')
    logging.info("Admin user initialized (if it didn't exist already)")

@user_bp.route('/register', methods=['POST'])
def register():
    """Register a new user."""
    data = request.get_json()
    
    if not data:
        return jsonify({
            'success': False,
            'error': 'No data provided'
        }), 400
    
    username = data.get('username')
    password = data.get('password')
    email = data.get('email')
    
    if not username or not password:
        return jsonify({
            'success': False,
            'error': 'Username and password are required'
        }), 400
    
    # Only admins can create admin users
    is_admin = data.get('is_admin', False)
    if is_admin and (not session.get('username') or not session.get('is_admin')):
        is_admin = False
    
    result = user_service.register(username, password, email, is_admin)
    
    if result['success']:
        return jsonify(result)
    else:
        return jsonify(result), 400

@user_bp.route('/login', methods=['POST'])
def login():
    """Authenticate a user."""
    data = request.get_json()
    
    if not data:
        return jsonify({
            'success': False,
            'error': 'No data provided'
        }), 400
    
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({
            'success': False,
            'error': 'Username and password are required'
        }), 400
    
    result = user_service.login(username, password)
    
    if result['success']:
        # Set session data
        session['username'] = username
        session['user_id'] = result['user']['id']
        session['is_admin'] = result['user']['is_admin']
        
        return jsonify(result)
    else:
        return jsonify(result), 401

@user_bp.route('/logout', methods=['POST'])
def logout():
    """Log out the current user."""
    session.clear()
    return jsonify({
        'success': True,
        'message': 'Logged out successfully'
    })

@user_bp.route('/me', methods=['GET'])
def get_current_user():
    """Get the current user's information."""
    if 'username' not in session:
        return jsonify({
            'success': False,
            'error': 'Not authenticated'
        }), 401
    
    user_info = user_service.get_user(session['username'])
    
    if user_info:
        return jsonify({
            'success': True,
            'user': user_info
        })
    else:
        session.clear()
        return jsonify({
            'success': False,
            'error': 'User not found'
        }), 404

@user_bp.route('/profile', methods=['PUT'])
def update_profile():
    """Update the current user's profile."""
    if 'username' not in session:
        return jsonify({
            'success': False,
            'error': 'Not authenticated'
        }), 401
    
    data = request.get_json()
    
    if not data:
        return jsonify({
            'success': False,
            'error': 'No data provided'
        }), 400
    
    # Only allow updating certain fields
    update_data = {}
    
    if 'email' in data:
        update_data['email'] = data['email']
    
    if 'password' in data:
        update_data['password'] = data['password']
    
    # Only admins can change is_admin status
    if 'is_admin' in data and session.get('is_admin'):
        update_data['is_admin'] = data['is_admin']
    
    result = user_service.update_user(session['username'], update_data)
    
    if result['success']:
        # Update session if needed
        if 'is_admin' in update_data:
            session['is_admin'] = update_data['is_admin']
            
        return jsonify(result)
    else:
        return jsonify(result), 400

@user_bp.route('/users', methods=['GET'])
def list_users():
    """List all users (admin only)."""
    if not session.get('is_admin'):
        return jsonify({
            'success': False,
            'error': 'Admin privileges required'
        }), 403
    
    users = user_service.list_users()
    return jsonify({
        'success': True,
        'users': users
    })

@user_bp.route('/users/<username>', methods=['DELETE'])
def delete_user(username):
    """Delete a user (admin only)."""
    if not session.get('is_admin'):
        return jsonify({
            'success': False,
            'error': 'Admin privileges required'
        }), 403
    
    # Prevent deleting your own account
    if username == session.get('username'):
        return jsonify({
            'success': False,
            'error': 'Cannot delete your own account'
        }), 400
    
    result = user_service.delete_user(username)
    
    if result['success']:
        return jsonify(result)
    else:
        return jsonify(result), 404 