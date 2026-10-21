from flask import Flask, render_template, request
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


@app.after_request
def support_https_preview_cookies(response):
    """Ensure session cookies work inside cross-origin HTTPS preview iframes as well as local HTTP."""
    is_https = request.is_secure or request.headers.get("X-Forwarded-Proto", "").lower() == "https"
    if is_https and "Set-Cookie" in response.headers:
        updated = []
        for cookie in response.headers.getlist("Set-Cookie"):
            if "SameSite=" not in cookie:
                cookie += "; SameSite=None"
            else:
                cookie = cookie.replace("SameSite=Lax", "SameSite=None")
            if "; Secure" not in cookie:
                cookie += "; Secure"
            updated.append(cookie)
        response.headers.setlist("Set-Cookie", updated)
    return response


@app.route("/")
def landing():
    return render_template("landing.html")


def open_browser():
    try:
        webbrowser.open("http://127.0.0.1:5000")
    except Exception:
        pass


if __name__ == "__main__":
    check_and_create_db()
    os.makedirs(Config.AUDIO_FOLDER, exist_ok=True)
    if Config.DEBUG and not os.environ.get("NO_OPEN_BROWSER"):
        threading.Timer(1.5, open_browser).start()
    app.run(host="0.0.0.0", debug=Config.DEBUG, port=5000)