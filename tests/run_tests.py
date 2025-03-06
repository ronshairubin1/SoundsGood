#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test runner for the unified feature extraction implementation.

This script runs all tests related to the unified feature extraction implementation:
1. Feature extraction tests
2. Webapp integration tests
3. Codebase consistency tests
"""

import os
import sys
import logging
import subprocess
import time
import argparse

# Add the current directory to the Python path to allow imports from 'backend'
# This is important for module imports in tests
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tests/output/test_runner.log"),
        logging.StreamHandler()
    ]
)

def run_test(test_path, description):
    """Run a test script and report results."""
    logging.info(f"Running {description} test: {test_path}")
    print(f"\n{'='*80}")
    print(f"RUNNING: {description}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    try:
        # Add the current directory to PYTHONPATH for the subprocess
        env = os.environ.copy()
        env['PYTHONPATH'] = os.path.abspath(os.path.dirname(__file__)) + os.pathsep + env.get('PYTHONPATH', '')
        
        result = subprocess.run([sys.executable, test_path], 
                              capture_output=True, 
                              text=True,
                              env=env)
        
        # Print output
        if result.stdout:
            print(result.stdout)
        
        # Print errors if any
        if result.stderr:
            print("ERRORS:")
            print(result.stderr)
        
        # Check if test passed
        if result.returncode == 0:
            elapsed = time.time() - start_time
            print(f"\n✅ {description} tests PASSED in {elapsed:.2f} seconds")
            logging.info(f"{description} tests passed in {elapsed:.2f} seconds")
            return True
        else:
            elapsed = time.time() - start_time
            print(f"\n❌ {description} tests FAILED in {elapsed:.2f} seconds")
            logging.error(f"{description} tests failed with code {result.returncode}")
            return False
            
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ {description} tests ERROR: {str(e)}")
        logging.error(f"Error running {description} test: {str(e)}")
        return False

def main():
    """Main function to run all tests."""
    parser = argparse.ArgumentParser(description='Run unified feature extraction tests')
    parser.add_argument('--feature-only', action='store_true', help='Run only feature extraction tests')
    parser.add_argument('--webapp-only', action='store_true', help='Run only webapp integration tests')
    parser.add_argument('--test', default=None, help='Run a specific test file')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs("tests/output", exist_ok=True)
    
    # Track test results
    results = []
    
    # Run a specific test if specified
    if args.test:
        if os.path.exists(args.test):
            success = run_test(args.test, f"Specific test: {args.test}")
            results.append((args.test, success))
        else:
            logging.error(f"Test file not found: {args.test}")
            print(f"❌ Test file not found: {args.test}")
            return 1
        
    # Run all tests
    else:
        # Define test suite
        test_suite = []
        
        # Feature extraction tests
        if not args.webapp_only:
            test_suite.extend([
                ("tests/feature_extraction/test_unified_extractor.py", "Unified Feature Extractor"),
                ("tests/feature_extraction/test_feature_unification.py", "Feature Unification"),
                ("backend/test_feature_unification.py", "Backend Feature Unification")
            ])
        
        # Webapp integration tests
        if not args.feature_only:
            test_suite.extend([
                ("tests/webapp/test_webapp_feature_extractor.py", "Webapp Feature Extractor Integration")
            ])
        
        # Run tests
        for test_path, description in test_suite:
            if os.path.exists(test_path):
                success = run_test(test_path, description)
                results.append((description, success))
            else:
                logging.warning(f"Test file not found: {test_path}")
                print(f"⚠️ Test file not found: {test_path}")
    
    # Report overall results
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    total_tests = len(results)
    passed_tests = sum(1 for _, success in results if success)
    
    for description, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} - {description}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    # Return exit code based on test results
    return 0 if passed_tests == total_tests else 1

if __name__ == "__main__":
    sys.exit(main()) 