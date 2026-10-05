from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "nhanes_final_xgboost_10_features.joblib"
)

model = joblib.load(MODEL_PATH)

FEATURES = [
    "age",
    "bun",
    "systolic_bp",
    "glucose",
    "diabetes",
    "uric_acid",
    "waist",
    "albumin",
    "bmi",
    "total_cholesterol"
]


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    probability = None
    error = None

    if request.method == "POST":
        try:
            data = {}

            for feature in FEATURES:
                value = request.form.get(feature)

                if value == "":
                    value = None
                else:
                    value = float(value)

                data[feature] = value

            input_data = pd.DataFrame(
                [data],
                columns=FEATURES
            )

            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1] * 100

            if prediction == 1:
                result = "CKD Detected"
            else:
                result = "No CKD Detected"

            probability = round(probability, 2)

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        result=result,
        probability=probability,
        error=error
    )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )