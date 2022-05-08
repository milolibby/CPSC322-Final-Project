import os
from flask import Flask, jsonify, request, render_template
from mysklearn import myutils
import csv

app  = Flask(__name__)

#homepage route
@app.route("/", methods=["GET"])
def index():
    return render_template("home_template.html"), 200#"<h1>Welcome to my app!!</h1>", 200 #"<h1>Welcome to my app!!</h1>", 200


@app.route("/rfc", methods=["GET"])
def predict():
    with open("./prediction.txt") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        prediction = []
        prediction.append(next(csv_reader))
        for i, row in enumerate(csv_reader):
            instance = []
            instance.append([att for att in row])
    return render_template("rfc_template.html",instance=instance, prediction=prediction)


def start_app():
    """allow other files to start the app"""
    #app = Flask(__name__)
    port = os.environ.get("PORT", 5000)
    app.run(debug=False, port=port,host="0.0.0.0")


if __name__ == "__main__":
    """start the app directly using this file"""
    port = os.environ.get("PORT", 5000)
    app.run(debug=False, port=port,host="0.0.0.0")