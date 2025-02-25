from flask import Flask, request
import logging

# Create a separate Flask instance for training
train_app = Flask(__name__)

@train_app.route('/train_model', methods=['POST'])
def train_model():
    """
    Example route that handles model training.
    """
    logging.info("Starting CNN training in separate train_app (no debug reloader).")
    # Training logic goes here, for example:
    # model.fit(...)
    logging.info("Training completed.")
    return "Training done!"

# If run directly, launch on port 5002, no debug, no reloader
if __name__ == "__main__":
    train_app.run(port=5002, debug=False, use_reloader=False)
