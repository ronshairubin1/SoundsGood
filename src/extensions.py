#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Flask extensions module.
This module provides a place to set up Flask extensions.
"""

# Create a mock SQLAlchemy db instance since it's imported in __init__.py
class MockDB:
    def init_app(self, app):
        pass

# Create a mock db instance
db = MockDB() 