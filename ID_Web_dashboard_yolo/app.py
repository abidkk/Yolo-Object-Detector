# from flask import Flask, render_template, jsonify
# from model import csv_data, seen_ids, start_tracking, stop_tracking
# from datetime import datetime
# import pandas as pd

# app = Flask(__name__)

# @app.route("/")
# def home():
#     now = datetime.now()
#     return render_template("index.html",
#                            time=now.strftime("%I:%M:%S %p"),
#                            date=now.strftime("%Y-%m-%d"),
#                            day=now.strftime("%A"))

# @app.route("/start", methods=["POST"])
# def start():
#     start_tracking()
#     return jsonify({"status": "started"})

# @app.route("/stop", methods=["POST"])
# def stop():
#     stop_tracking()
#     return jsonify({"status": "stopped"})

# @app.route('/summary')
# def summary():
#     df = pd.read_csv("object_counts.csv")
#     with_id = len(df[df["label"].isin(["person_with_id"])])
#     without_id = len(df[df["label"].isin(["person_without_id"])])
#     total_persons = with_id+without_id

   
#     summary_data = {
#         "Person_with_ID": int(with_id),
#         "Person_without_ID": int(without_id ),
#         "Total_Unique_Persons": int(total_persons )
#         }
    

#     return jsonify(summary_data)

# if __name__ == "__main__":
#     app.run(debug=False, threaded=True)



# app.py

from flask import Flask, render_template, jsonify
from model import csv_data, seen_ids, start_tracking, stop_tracking
from datetime import datetime
import pandas as pd

app = Flask(__name__)

# Route to render the homepage with current date, time, and day
@app.route("/")
def home():
    now = datetime.now()
    return render_template("index.html",
                           time=now.strftime("%I:%M:%S %p"),
                           date=now.strftime("%Y-%m-%d"),
                           day=now.strftime("%A"))

# API endpoint to start tracking via POST request
@app.route("/start", methods=["POST"])
def start():
    start_tracking()
    return jsonify({"status": "started"})

# API endpoint to stop tracking via POST request
@app.route("/stop", methods=["POST"])
def stop():
    stop_tracking()
    return jsonify({"status": "stopped"})

# Endpoint to return summary data in JSON format
@app.route('/summary')
def summary():
    # Read logged detection data
    df = pd.read_csv("object_counts.csv")

    # Count each class based on labels
    with_id = len(df[df["label"].isin(["person_with_id"])])
    without_id = len(df[df["label"].isin(["person_without_id"])])
    bikes = len(df[df["label"].isin(["bike"])])
    cars = len(df[df["label"].isin(["car"])])

    total_objects = with_id + without_id + bikes + cars

    # Create summary dictionary
    summary_data = {
        "Person_with_ID": with_id,
        "Person_without_ID": without_id,
        "Bike": bikes,
        "Car": cars,
        "Total_Objects": total_objects
    }

    return jsonify(summary_data)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=False, threaded=True)
