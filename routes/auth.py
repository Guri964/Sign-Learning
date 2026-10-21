from flask import Blueprint, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from database import get_connection

auth = Blueprint("auth", __name__)

@auth.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()
        role = request.form.get("role", "").strip()
        if not all([full_name, email, password, confirm_password, role]):
            flash("All fields are required.", "error")
            return render_template("signup.html")
        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return render_template("signup.html")
        if role not in ["Student", "Interpreter"]:
            flash("Invalid role selected.", "error")
            return render_template("signup.html")
        hashed = generate_password_hash(password)
        conn = get_connection()
        try:
            conn.execute(
                "INSERT INTO users (full_name, email, password, role) VALUES (?, ?, ?, ?)",
                (full_name, email, hashed, role)
            )
            conn.commit()
            flash("Account created. Please log in.", "success")
            return redirect(url_for("auth.login"))
        except Exception:
            flash("Email already registered.", "error")
            return render_template("signup.html")
        finally:
            conn.close()
    return render_template("signup.html")

@auth.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        if not email or not password:
            flash("Email and password are required.", "error")
            return render_template("login.html")
        conn = get_connection()
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        conn.close()
        if user and check_password_hash(user["password"], password):
            session["user_id"] = user["id"]
            session["full_name"] = user["full_name"]
            session["role"] = user["role"]
            session["language"] = user["language_preference"]
            return redirect(url_for("dashboard.index"))
        flash("Invalid email or password.", "error")
        return render_template("login.html")
    return render_template("login.html")

@auth.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("landing"))