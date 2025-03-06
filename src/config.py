#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file re-exports the Config class from the root config.py file
to maintain compatibility with imports that expect src.config.
"""

import sys
import os
# Add the root directory to the path so we can import from root config.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the Config class from the root config.py
from config import Config

# Export the Config class
__all__ = ['Config'] 