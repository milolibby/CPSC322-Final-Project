import os
import pickleshare
from flask import Flask, jsonify, request, render_template
from mysklearn.myrandomforestclassifier import MyRandomForestClassifier
from mysklearn import myutils
import csv

app  = Flask(__name__)

#homepage route
@app.route("/", methods=["GET"])
def index():
    return render_template("home_template.html"), 200#"<h1>Welcome to my app!!</h1>", 200 #"<h1>Welcome to my app!!</h1>", 200


@app.route("/rfc", methods=["GET"])
def predict():
    """
    N = 20
    M = 7
    F = 2
    nba_dataset = myutils.get_nba_ratings_dataset()

    #strip the classification and append it to y_labels
    y_labels = [row[-1] for row in nba_dataset]
    nba_dataset = [row[0:-1] for row in nba_dataset]

    X_train, X_test, y_train, y_test = myutils.get_train_test_sets(nba_dataset, y_labels)
    my_rfc = MyRandomForestClassifier(N,M,F)
    my_rfc.fit(X_train,y_train)
    my_prediction = my_rfc.predict(X_test)
    print()
    """
    with open("./prediction.txt") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        prediction = []
        prediction.append(next(csv_reader))
        for i, row in enumerate(csv_reader):
            instance = []
            instance.append([att for att in row])

    #test_instance = ["young",  "high" , "average" , "high" , "efficient", "low volume", "decent", "high volume"]

    #prediction = ["special"]
    prediction_dict = {"test_instance":instance, "prediction": prediction}

    #return jsonify(prediction_dict), 200#"<h1>"+thing+"</h1>", 200 
    #return "<h1>"+f"asdf{jsonify(prediction_dict)}"+"<h1>"
    return render_template("rfc_template.html",instance=instance, prediction=prediction)



if __name__ == "__main__":
    #predict()
    port = os.environ.get("PORT", 5000)
    app.run(debug=True, port=port,host="0.0.0.0") #TODO change dubug to False