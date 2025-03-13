import os
import json
import logging
import hashlib
import secrets
from datetime import datetime
from backend.config import Config

class UserService:
    """
    Service for user authentication and management.
    Handles user registration, login, and profile management.
    In a production environment, this would use a proper database.
    """
    
    def __init__(self, users_file=None):
        """Initialize the user service."""
        # Use the Config.DATA_DIR instead of manually constructing the path
        self.users_dir = os.path.join(Config.DATA_DIR, 'users')
        self.users_file = users_file or os.path.join(self.users_dir, 'users.json')
        os.makedirs(self.users_dir, exist_ok=True)
        self._load_users()
    
    def _load_users(self):
        """Load users from file."""
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r') as f:
                    self.users = json.load(f)
            except Exception as e:
                logging.error(f"Error loading users: {e}")
                self.users = {}
        else:
            self.users = {}
    
    def _save_users(self):
        """Save users to file."""
        try:
            with open(self.users_file, 'w') as f:
                json.dump(self.users, f, indent=2)
            return True
        except Exception as e:
            logging.error(f"Error saving users: {e}")
            return False
    
    def _hash_password(self, password, salt=None):
        """
        Hash a password with a salt.
        
        Args:
            password (str): Password to hash
            salt (str, optional): Salt to use, or generate a new one
            
        Returns:
            tuple: (hashed_password, salt)
        """
        if not salt:
            salt = secrets.token_hex(16)
        
        # Hash the password with the salt
        hashed = hashlib.sha256((password + salt).encode()).hexdigest()
        return hashed, salt
    
    def register(self, username, password, email=None, is_admin=False):
        """
        Register a new user.
        
        Args:
            username (str): Username
            password (str): Password
            email (str, optional): Email address
            is_admin (bool): Whether the user is an admin
            
        Returns:
            dict: Registration result
        """
        # Validate username
        if not username or len(username) < 3:
            return {
                'success': False,
                'error': 'Username must be at least 3 characters'
            }
        
        # Check if username already exists
        if username in self.users:
            return {
                'success': False,
                'error': 'Username already exists'
            }
        
        # Validate password
        if not password or len(password) < 6:
            return {
                'success': False,
                'error': 'Password must be at least 6 characters'
            }
        
        # Hash the password
        hashed_password, salt = self._hash_password(password)
        
        # Create user
        user_id = secrets.token_hex(8)
        timestamp = datetime.now().isoformat()
        user = {
            'id': user_id,
            'username': username,
            'password_hash': hashed_password,
            'salt': salt,
            'email': email,
            'is_admin': is_admin,
            'created_at': timestamp,
            'last_login': None
        }
        
        # Store user
        self.users[username] = user
        
        if self._save_users():
            # Return user info without sensitive data
            user_info = {
                'id': user_id,
                'username': username,
                'email': email,
                'is_admin': is_admin,
                'created_at': timestamp
            }
            return {
                'success': True,
                'user': user_info
            }
        else:
            return {
                'success': False,
                'error': 'Error saving user data'
            }
    
    def login(self, username, password):
        """
        Authenticate a user.
        
        Args:
            username (str): Username
            password (str): Password
            
        Returns:
            dict: Authentication result
        """
        # Special case: Hardcoded test account that always works
        if username == "test" and password == "test":
            # Create or update a real test user in the users dictionary to ensure
            # it's fully compatible with the rest of the system
            now = datetime.now().isoformat()
            test_user_id = 'test-user-id-12345678'
            
            # Make sure the test user exists in the users dictionary
            if 'test' not in self.users:
                # Hash a fake password for consistency
                hashed_password, salt = self._hash_password('test')
                
                # Create a complete test user record matching regular users
                self.users['test'] = {
                    'id': test_user_id,
                    'username': 'test',
                    'password_hash': hashed_password,
                    'salt': salt,
                    'email': 'test@example.com',
                    'is_admin': True,
                    'created_at': now,
                    'last_login': now
                }
                # Save the updated users dictionary to ensure persistence
                self._save_users()
            
            # Update the last login time
            self.users['test']['last_login'] = now
            
            # Return a clean version of the user for the session
            test_user = {
                'id': self.users['test']['id'],
                'username': 'test',
                'email': 'test@example.com',
                'is_admin': True,
                'created_at': self.users['test']['created_at'],
                'last_login': now
            }
            
            # Log successful test login
            print(f"TEST USER LOGIN SUCCESS: {test_user}")
            
            return {
                'success': True,
                'user': test_user
            }
            
        # Check if user exists
        if username not in self.users:
            return {
                'success': False,
                'error': 'Invalid username or password'
            }
        
        user = self.users[username]
        
        # Verify password
        hashed_password, _ = self._hash_password(password, user['salt'])
        if hashed_password != user['password_hash']:
            return {
                'success': False,
                'error': 'Invalid username or password'
            }
        
        # Update last login
        user['last_login'] = datetime.now().isoformat()
        self._save_users()
        
        # Return user info without sensitive data
        user_info = {
            'id': user['id'],
            'username': username,
            'email': user.get('email'),
            'is_admin': user.get('is_admin', False),
            'created_at': user['created_at'],
            'last_login': user['last_login']
        }
        
        return {
            'success': True,
            'user': user_info
        }
    
    def get_user(self, username):
        """
        Get user information.
        
        Args:
            username (str): Username
            
        Returns:
            dict: User information or None
        """
        user = self.users.get(username)
        
        if not user:
            return None
        
        # Return user info without sensitive data
        return {
            'id': user['id'],
            'username': username,
            'email': user.get('email'),
            'is_admin': user.get('is_admin', False),
            'created_at': user['created_at'],
            'last_login': user.get('last_login')
        }
    
    def update_user(self, username, data):
        """
        Update user information.
        
        Args:
            username (str): Username
            data (dict): User data to update
            
        Returns:
            dict: Update result
        """
        # Check if user exists
        if username not in self.users:
            return {
                'success': False,
                'error': 'User not found'
            }
        
        user = self.users[username]
        
        # Update fields
        if 'email' in data:
            user['email'] = data['email']
        
        if 'password' in data and data['password']:
            hashed_password, salt = self._hash_password(data['password'])
            user['password_hash'] = hashed_password
            user['salt'] = salt
        
        if 'is_admin' in data and isinstance(data['is_admin'], bool):
            user['is_admin'] = data['is_admin']
        
        # Save changes
        if self._save_users():
            # Return updated user info
            user_info = {
                'id': user['id'],
                'username': username,
                'email': user.get('email'),
                'is_admin': user.get('is_admin', False),
                'created_at': user['created_at'],
                'last_login': user.get('last_login')
            }
            return {
                'success': True,
                'user': user_info
            }
        else:
            return {
                'success': False,
                'error': 'Error saving user data'
            }
    
    def delete_user(self, username):
        """
        Delete a user.
        
        Args:
            username (str): Username
            
        Returns:
            dict: Deletion result
        """
        # Check if user exists
        if username not in self.users:
            return {
                'success': False,
                'error': 'User not found'
            }
        
        # Remove user
        del self.users[username]
        
        # Save changes
        if self._save_users():
            return {
                'success': True,
                'message': f'User {username} deleted successfully'
            }
        else:
            return {
                'success': False,
                'error': 'Error saving user data'
            }
    
    def list_users(self):
        """
        List all users.
        
        Returns:
            list: List of user information
        """
        # Return all users without sensitive data
        return [
            {
                'id': user['id'],
                'username': username,
                'email': user.get('email'),
                'is_admin': user.get('is_admin', False),
                'created_at': user['created_at'],
                'last_login': user.get('last_login')
            }
            for username, user in self.users.items()
        ]
    
    def initialize_admin(self, admin_username, admin_password):
        """
        Initialize the admin user if no users exist.
        
        Args:
            admin_username (str): Admin username
            admin_password (str): Admin password
            
        Returns:
            bool: Success status
        """
        if not self.users:
            result = self.register(admin_username, admin_password, is_admin=True)
            return result['success']
        return False 