from flask import Flask, render_template
from flask_cors import CORS
from config import Config
from database import check_and_create_db
from routes.auth import auth
from routes.dashboard import dashboard
from routes.translator import translator
from routes.tutor import tutor
import webbrowser
import threading
import os

app = Flask(__name__)
app.secret_key = Config.SECRET_KEY
app.config.from_object(Config)
CORS(app)

app.register_blueprint(auth)
app.register_blueprint(dashboard)
app.register_blueprint(translator)
app.register_blueprint(tutor)

@app.route("/")
def landing():
    return render_template("landing.html")

def open_browser():
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == "__main__":
    check_and_create_db()
    os.makedirs(Config.AUDIO_FOLDER, exist_ok=True)
    if Config.DEBUG:
        threading.Timer(1.5, open_browser).start()
    app.run(host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", 5000)), debug=Config.DEBUG)