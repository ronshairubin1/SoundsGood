#!/usr/bin/env python3
import os
import sys
import importlib

# Test imports that were previously from the src directory
try:
    print("Attempting to import backend.config...")
    from backend.config import Config
    print("SUCCESS: backend.config imported")
except Exception as e:
    print(f"ERROR importing backend.config: {e}")

try:
    print("\nAttempting to import backend.src.api.ml_api...")
    from backend.src.api.ml_api import MlApi
    print("SUCCESS: backend.src.api.ml_api imported")
except Exception as e:
    print(f"ERROR importing backend.src.api.ml_api: {e}")

try:
    print("\nAttempting to import backend.src.api.dictionary_api...")
    from backend.src.api.dictionary_api import dictionary_bp
    print("SUCCESS: backend.src.api.dictionary_api imported")
except Exception as e:
    print(f"ERROR importing backend.src.api.dictionary_api: {e}")

try:
    print("\nAttempting to import backend.src.routes.ml_routes...")
    mod = importlib.import_module('backend.src.routes.ml_routes')
    print("SUCCESS: backend.src.routes.ml_routes imported")
    if hasattr(mod, 'get_model_metadata_direct'):
        print("get_model_metadata_direct function found in module")
    else:
        print("get_model_metadata_direct function NOT found in module")
except Exception as e:
    print(f"ERROR importing backend.src.routes.ml_routes: {e}")

print("\nAll import tests complete.") 