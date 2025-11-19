from flask import Flask
from flask_cors import CORS
from app.routes.chatbot_routes import chat_bp

def create_app():
    app = Flask(__name__)
    ##CORS(app)

    app.register_blueprint(chat_bp, url_prefix="/")

    return app