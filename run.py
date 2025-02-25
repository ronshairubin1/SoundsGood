import os
import sys
import logging
from multiprocessing import Process

# Add the src directory to Python path
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

from app import app       # your main Flask app
from src.routes.train_app import train_app  
# import theseparate training app

def run_main_app():
    """
    Runs the main Flask app in debug mode on port 5001.
    """
    app.run(debug=True, port=5001, use_reloader=False)

def run_train_app():
    """
    Runs the separate training Flask app without debug mode on port 5002.
    """
    train_app.run(debug=False, port=5002, use_reloader=False)

def main():
    logging.basicConfig(level=logging.DEBUG)

    # Spawn the training app in a separate process
    p = Process(target=run_train_app)
    p.daemon = True
    p.start()

    # Run the main app in the main thread
    run_main_app()

if __name__ == "__main__":
    main()