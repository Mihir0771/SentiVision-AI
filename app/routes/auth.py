"""
Authentication blueprint — login, registration, logout, and profile management.
"""

from __future__ import annotations

import logging
from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user

from app.models.analysis import Analysis
from app.models.user import User

logger = logging.getLogger(__name__)

bp = Blueprint("auth", __name__, template_folder="../templates")


@bp.route("/login", methods=["GET", "POST"])
def login():
    """Handle user login."""
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))

    if request.method == "POST":
        identifier = request.form.get("identifier", "").strip()
        password = request.form.get("password", "")
        remember = bool(request.form.get("remember"))

        if not identifier or not password:
            flash("Please enter both username/email and password.", "error")
            return render_template("login.html")

        db = getattr(current_app, "db", None)
        if db is None:
            flash("Database service currently unavailable. Please try again shortly.", "warning")
            return render_template("login.html")

        # Try lookup by email first, then username
        user = User.find_by_email(db, identifier) or User.find_by_username(db, identifier)
        if user and user.check_password(password):
            login_user(user, remember=remember)
            flash(f"Welcome back, {user.username}!", "success")
            next_page = request.args.get("next")
            return redirect(next_page or url_for("main.dashboard"))
        else:
            flash("Invalid credentials. Please verify your login details.", "error")

    return render_template("login.html")


@bp.route("/register", methods=["GET", "POST"])
def register():
    """Handle new user registration."""
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        # Validation
        if not username or not email or not password:
            flash("All fields are required.", "error")
            return render_template("register.html")

        if len(username) < 3:
            flash("Username must be at least 3 characters long.", "error")
            return render_template("register.html")

        if len(password) < 6:
            flash("Password must be at least 6 characters long.", "error")
            return render_template("register.html")

        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return render_template("register.html")

        db = getattr(current_app, "db", None)
        if db is None:
            flash("Database service currently unavailable. Please try again shortly.", "warning")
            return render_template("register.html")

        # Check existing user
        if User.find_by_username(db, username):
            flash("Username is already taken. Please choose another.", "error")
            return render_template("register.html")

        if User.find_by_email(db, email):
            flash("Email is already registered. Please sign in instead.", "error")
            return render_template("register.html")

        try:
            new_user = User(username=username, email=email)
            new_user.set_password(password)
            new_user.save(db)

            login_user(new_user)
            flash("Registration successful! Welcome to SentiVision AI.", "success")
            return redirect(url_for("main.dashboard"))
        except Exception as e:
            logger.error("Error creating user: %s", e)
            flash("An unexpected error occurred during registration. Please try again.", "error")

    return render_template("register.html")


@bp.route("/logout")
@login_required
def logout():
    """Log out the current user."""
    logout_user()
    flash("You have been logged out successfully.", "info")
    return redirect(url_for("auth.login"))


@bp.route("/profile")
@login_required
def profile():
    """Display user profile and account statistics."""
    db = getattr(current_app, "db", None)
    user_stats = {}
    if db is not None:
        try:
            user_stats = Analysis.get_user_stats(db, current_user.id)
        except Exception as e:
            logger.warning("Error fetching user profile stats: %s", e)

    return render_template("profile.html", user=current_user, stats=user_stats)
