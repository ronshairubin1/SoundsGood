from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Main App (Debug Mode)."

if __name__ == "__main__":
    # Run this app on port 5001 with debug mode on
    app.run(port=5001, debug=True, use_reloader=True)
