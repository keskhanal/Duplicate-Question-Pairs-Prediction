#importing all the necessary libraries
from flask import render_template

from flask_app import app

@app.route("/", methods=["POST", "GET"])
def index():
    return render_template("index.html", name=index)
