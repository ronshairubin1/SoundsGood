#!/usr/bin/env python3
import os
import sys
import subprocess
import signal
import time
import platform
import webbrowser
import threading

def check_and_kill_port(port=5002):
    """
    Check if the specified port is in use and kill the process if it is
    """
    print(f"Checking if port {port} is in use...")
    
    # Different command based on OS
    system = platform.system().lower()
    
    if system == 'darwin' or system == 'linux':  # macOS or Linux
        # Find PID using the port
        find_pid_cmd = f"lsof -i tcp:{port} -t"
        try:
            pid_output = subprocess.check_output(find_pid_cmd, shell=True).decode().strip()
            if pid_output:
                # Split the output in case multiple PIDs are returned
                pids = pid_output.split('\n')
                print(f"Found {len(pids)} process(es) using port {port}")
                
                for pid in pids:
                    pid = pid.strip()
                    if pid:
                        print(f"Terminating process with PID {pid}...")
                        try:
                            # Kill the process
                            os.kill(int(pid), signal.SIGTERM)
                            time.sleep(0.5)  # Give it time to terminate
                        except Exception as e:
                            print(f"Error terminating process {pid}: {e}")
                
                print(f"All processes terminated.")
                return True
        except subprocess.CalledProcessError:
            # No process found using the port
            pass
    elif system == 'windows':
        # For Windows
        find_pid_cmd = f"netstat -ano | findstr :{port}"
        try:
            output = subprocess.check_output(find_pid_cmd, shell=True).decode()
            if output:
                # Extract PID from the output
                for line in output.splitlines():
                    if f":{port}" in line and "LISTENING" in line:
                        pid = line.strip().split()[-1]
                        print(f"Found process using port {port}: PID {pid}")
                        print(f"Terminating process...")
                        # Kill the process
                        subprocess.call(f"taskkill /F /PID {pid}", shell=True)
                        time.sleep(0.5)  # Give it time to terminate
                        print(f"Process terminated.")
                        return True
        except subprocess.CalledProcessError:
            # No process found using the port
            pass
    
    print(f"No process found using port {port}.")
    return False

def open_browser(port=5002):
    """
    Open the browser to the application URL after a short delay
    """
    # Direct to the auto-logout route to ensure the user starts logged out
    url = f"http://localhost:{port}/auto-logout"
    print(f"Opening browser to {url}...")
    
    # Give the server more time to start up
    time.sleep(3)
    
    # Try to open the browser with better error handling
    try:
        # Try to open with the default browser
        success = webbrowser.open(url)
        
        if not success:
            print("Failed to open with default browser, trying specific browsers...")
            # Try specific browsers by name
            for browser in ['chrome', 'firefox', 'safari']:
                try:
                    browser_controller = webbrowser.get(browser)
                    browser_controller.open(url)
                    print(f"Opened URL with {browser}")
                    break
                except Exception:
                    continue
            else:
                # If all browsers failed, try system-specific command
                print("Trying system command to open browser...")
                system = platform.system().lower()
                if system == 'darwin':  # macOS
                    subprocess.run(['open', url], check=False)
                elif system == 'windows':
                    subprocess.run(['start', url], shell=True, check=False)
                elif system == 'linux':
                    subprocess.run(['xdg-open', url], check=False)
                else:
                    print("WARNING: Could not open browser automatically.")
                    print(f"Please open {url} manually in your browser.")
    except Exception as e:
        print(f"Error trying to open browser: {e}")
        print(f"Please open {url} manually in your browser.")

def main():
    """
    Run the Sound Classifier application
    - Checks if we're already in a virtual environment
    - If not, respawns this script inside the virtual environment
    - Once in a virtual environment, runs main.py
    """
    # First, check if port 5002 is in use and kill the process if necessary
    check_and_kill_port(5002)
    
    # Check if we're already in a virtual environment
    in_venv = sys.prefix != sys.base_prefix

    if not in_venv:
        print("Activating virtual environment...")
        
        # Path to the virtual environment's activate script
        venv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv")
        activate_script = os.path.join(venv_path, "bin", "activate")
        
        if not os.path.exists(activate_script):
            print(f"Error: Virtual environment not found at {venv_path}")
            print("Please create a virtual environment first: python -m venv venv")
            sys.exit(1)
        
        # Command to activate venv and rerun this script
        cmd = f"source {activate_script} && python {__file__}"
        
        # Use subprocess to run the command in a shell
        process = subprocess.Popen(cmd, shell=True)
        process.communicate()
        sys.exit(process.returncode)
    else:
        print("Virtual environment already active")
        
        # Now run the main application
        print("Starting Sound Classifier application...")
        main_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")
        
        # Start a thread to open the browser after a short delay
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        try:
            # Run the main script directly as a subprocess
            # This ensures it runs in the foreground and doesn't terminate
            subprocess.call([sys.executable, main_script])
        except Exception as e:
            print(f"Error running main application: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()