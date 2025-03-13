#!/usr/bin/env python3
import os
import sys
import subprocess
import signal
import time
import platform
import webbrowser
import threading
import logging
import traceback
from backend.config import Config

# Set up logging
logging.basicConfig(
    filename=os.path.join(Config.LOGS_DIR, 'app_startup.log'),
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add a stream handler to also print to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

logger = logging.getLogger('run_script')

def check_and_kill_port(port=5002):
    """
    Check if the specified port is in use and kill the process if it is.
    Attempts with SIGTERM first, then SIGKILL if needed.
    """
    logger.info(f"Checking if port {port} is in use...")
    
    # Different command based on OS
    system = platform.system().lower()
    logger.debug(f"Detected operating system: {system}")
    
    def is_port_in_use():
        """Helper function to check if port is in use"""
        if system == 'darwin' or system == 'linux':
            cmd = f"lsof -i tcp:{port} -t"
        elif system == 'windows':
            cmd = f"netstat -ano | findstr :{port} | findstr LISTENING"
        else:
            return False
            
        try:
            output = subprocess.check_output(cmd, shell=True).decode().strip()
            return bool(output)
        except subprocess.CalledProcessError:
            return False
    
    # Maximum attempts to kill processes
    max_attempts = 3
    
    for attempt in range(max_attempts):
        if not is_port_in_use():
            logger.info(f"Port {port} is free.")
            return True
            
        logger.info(f"Attempt {attempt + 1}/{max_attempts} to free port {port}")
        
        if system == 'darwin' or system == 'linux':  # macOS or Linux
            # Find PID using the port
            find_pid_cmd = f"lsof -i tcp:{port} -t"
            logger.debug(f"Running command: {find_pid_cmd}")
            try:
                pid_output = subprocess.check_output(find_pid_cmd, shell=True).decode().strip()
                if pid_output:
                    # Split the output in case multiple PIDs are returned
                    pids = pid_output.split('\n')
                    logger.info(f"Found {len(pids)} process(es) using port {port}")
                    
                    for pid in pids:
                        pid = pid.strip()
                        if pid:
                            logger.info(f"Terminating process with PID {pid}...")
                            try:
                                # Try SIGTERM first
                                if attempt < 1:
                                    os.kill(int(pid), signal.SIGTERM)
                                    logger.debug(f"Sent SIGTERM to PID {pid}")
                                else:
                                    # Use SIGKILL for subsequent attempts
                                    os.kill(int(pid), signal.SIGKILL)
                                    logger.debug(f"Sent SIGKILL to PID {pid}")
                                
                                time.sleep(1.0)  # Give it more time to terminate
                            except Exception as e:
                                logger.error(f"Error terminating process {pid}: {e}")
            except subprocess.CalledProcessError:
                # No process found using the port
                logger.debug("No process found using the port (CalledProcessError)")
                pass
        elif system == 'windows':
            # For Windows
            find_pid_cmd = f"netstat -ano | findstr :{port}"
            logger.debug(f"Running command: {find_pid_cmd}")
            try:
                output = subprocess.check_output(find_pid_cmd, shell=True).decode()
                if output:
                    # Extract PID from the output
                    for line in output.splitlines():
                        if f":{port}" in line and "LISTENING" in line:
                            pid = line.strip().split()[-1]
                            logger.info(f"Found process using port {port}: PID {pid}")
                            logger.info(f"Terminating process...")
                            # Kill the process
                            kill_cmd = f"taskkill {'/F' if attempt > 0 else ''} /PID {pid}"
                            logger.debug(f"Running command: {kill_cmd}")
                            subprocess.call(kill_cmd, shell=True)
                            time.sleep(1.0)  # Give it more time to terminate
            except subprocess.CalledProcessError:
                # No process found using the port
                logger.debug("No process found using the port (CalledProcessError)")
                pass
    
    # Final check
    if is_port_in_use():
        logger.error(f"Failed to free port {port} after {max_attempts} attempts.")
        return False
    else:
        logger.info(f"Successfully freed port {port}.")
        return True

def open_browser(port=5002, delay=1.5):
    """Open browser to the application after a short delay"""
    try:
        time.sleep(delay)  # Give the web app time to start
        url = f"http://localhost:{port}/test-login"
        logger.info(f"Opening browser to {url}...")
        webbrowser.open(url)
    except Exception as e:
        logger.error(f"Error opening browser: {e}")

def main():
    """
    Run the Sound Classifier application
    - Checks if we're already in a virtual environment
    - If not, respawns this script inside the virtual environment
    - Once in a virtual environment, runs main.py
    """
    logger.info("Starting Sound Classifier launcher")
    
    # First, check if port 5002 is in use and kill the process if necessary
    check_and_kill_port(5002)
    
    # Check if we're already in a virtual environment
    in_venv = sys.prefix != sys.base_prefix
    logger.info(f"Running in virtual environment: {in_venv}")
    logger.debug(f"sys.prefix: {sys.prefix}")
    logger.debug(f"sys.base_prefix: {sys.base_prefix}")

    if not in_venv:
        logger.info("Activating virtual environment...")
        
        # Path to the virtual environment's activate script
        venv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv")
        activate_script = os.path.join(venv_path, "bin", "activate")
        logger.debug(f"Virtual environment path: {venv_path}")
        logger.debug(f"Activate script path: {activate_script}")
        
        if not os.path.exists(activate_script):
            logger.error(f"Error: Virtual environment not found at {venv_path}")
            logger.error("Please create a virtual environment first: python -m venv venv")
            sys.exit(1)
        
        # Command to activate venv and rerun this script
        cmd = f"source {activate_script} && python {__file__}"
        logger.debug(f"Running command: {cmd}")
        
        # Use subprocess to run the command in a shell
        logger.info("Restarting script in virtual environment")
        process = subprocess.Popen(cmd, shell=True)
        process.communicate()
        logger.debug(f"Subprocess exited with code: {process.returncode}")
        sys.exit(process.returncode)
    else:
        logger.info("Virtual environment already active")
        
        # Now run the main application
        logger.info("Starting Sound Classifier application...")
        main_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")
        logger.info(f"Main script path: {main_script}")
        
        # Start a thread to open the browser after a short delay
        logger.debug("Starting browser thread")
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        try:
            # Check if command line arguments are provided
            if len(sys.argv) > 1 and sys.argv[1] == "train":
                # Run the training command with necessary arguments
                training_args = [
                    sys.executable,
                    main_script,
                    "train",              # Command to train a model
                    "--dict_name", "EhOh",  # Dictionary name (using EhOh specifically)
                    "--model_type", "cnn",  # Model type (CNN)
                    "--clear_cache", "true"  # Clear cache flag (force regeneration)
                ]
                logger.info(f"Executing training: {' '.join(training_args)}")
                subprocess.call(training_args)
            else:
                # Run the web application
                logger.info("Starting web application...")
                # Run main.py with no arguments to start the web application (default behavior)
                web_args = [sys.executable, main_script]
                logger.info(f"Executing web app: {' '.join(web_args)}")
                try:
                    output = subprocess.check_output(web_args, stderr=subprocess.STDOUT)
                    logger.info("Web app output: %s", output.decode())
                except subprocess.CalledProcessError as e:
                    logger.error("Web app error (code %d): %s", e.returncode, e.output.decode())
                    print("Error running main.py:", e.output.decode())
        except Exception as e:
            logger.error(f"Error running application: {e}")
            logger.error(traceback.format_exc())
            sys.exit(1)

if __name__ == "__main__":
    main()