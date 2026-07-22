from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load Model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "dropout_prediction_model.pkl")

model = joblib.load(MODEL_PATH)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get values from HTML form
    age = int(request.form["Age"])

    gender = request.form["Gender"]
    gender = 1 if gender == "Male" else 0

    year = int(request.form["Year_of_Study"])

    attendance = float(request.form["Attendance_Percent"])

    study_hours = float(request.form["Study_Hours_Per_Day"])

    previous_gpa = float(request.form["Previous_GPA"])

    backlogs = int(request.form["Backlogs"])

    financial_stress = int(request.form["Financial_Stress_Score"])

    stress_level = int(request.form["Stress_Level"])

    burnout = request.form["Burnout_Level"]

    burnout_mapping = {
        "Low": 0,
        "Medium": 1,
        "High": 2
    }

    burnout = burnout_mapping[burnout]

    # Create input array
    input_data = np.array([[
        age,
        gender,
        year,
        attendance,
        study_hours,
        previous_gpa,
        backlogs,
        financial_stress,
        stress_level,
        burnout
    ]])

    # Prediction
    prediction = model.predict(input_data)[0]

    # Confidence
    confidence = round(max(model.predict_proba(input_data)[0]) * 100, 2)

    if prediction == 1:
        result = "⚠️ Student Will Drop Out"
    else:
        result = "✅ Student Will Not Drop Out"

    return render_template(
        "index.html",
        prediction=result,
        confidence=confidence
    )


if __name__ == "__main__":
    if __name__ == "__main__":
        port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)