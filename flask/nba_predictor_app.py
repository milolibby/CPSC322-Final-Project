import os
import pickleshare
from flask import Flask, jsonify, request, render_template

app  = Flask(__name__)

#homepage route
@app.route("/", methods=["GET"])
def index():
    return render_template("home_template.html"), 200#"<h1>Welcome to my app!!</h1>", 200


@app.route("/rfc", methods=["GET"])
def predict():
    return "<h1>Return the results of the myrandomforestclassifier here!</h1>", 200 



if __name__ == "__main__":
    port = os.environ.get("PORT", 5000)
    app.run(debug=True, port=port,host="0.0.0.0") #TODO change dubu to False