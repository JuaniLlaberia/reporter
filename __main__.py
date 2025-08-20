import logging
from dotenv import load_dotenv
from flask import Flask
from src.routes.reporter import reporter_bp

load_dotenv()

def create_app():
    app = Flask(__name__)
    app.register_blueprint(reporter_bp)

    return app

logging.basicConfig(filename='reporter.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

def main():
    """
    Main entry point for the Flask reporter API.
    Starts the web server to create reports via HTTP POST requests.
    """
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5000)

if __name__ == "__main__":
    main()