
import os
import pickle
import numpy as np

from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "naive.pkl")

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


# HTML + CSS + Flask template in one file
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Employee Attrition Predictor</title>

    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #eef2ff, #f8fafc);
            color: #172033;
            min-height: 100vh;
            padding: 35px 16px;
        }

        .container {
            max-width: 1000px;
            margin: auto;
        }

        .header {
            text-align: center;
            margin-bottom: 28px;
        }

        .logo {
            display: inline-block;
            background: #e0e7ff;
            color: #4338ca;
            padding: 10px 16px;
            border-radius: 30px;
            font-size: 13px;
            font-weight: bold;
            margin-bottom: 18px;
        }

        h1 {
            font-size: clamp(28px, 5vw, 42px);
            letter-spacing: -1px;
            margin-bottom: 12px;
        }

        .subtitle {
            color: #64748b;
            line-height: 1.7;
            max-width: 620px;
            margin: auto;
        }

        .card {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 22px;
            padding: 32px;
            box-shadow: 0 18px 50px rgba(15, 23, 42, 0.07);
        }

        .section-title {
            font-size: 21px;
            margin-bottom: 8px;
        }

        .section-description {
            color: #64748b;
            font-size: 14px;
            line-height: 1.6;
            margin-bottom: 28px;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 22px;
        }

        .field {
            display: flex;
            flex-direction: column;
            gap: 9px;
        }

        label {
            font-size: 14px;
            font-weight: 600;
            color: #334155;
        }

        input, select {
            width: 100%;
            padding: 14px;
            border: 1px solid #cbd5e1;
            border-radius: 11px;
            background: #fff;
            color: #172033;
            font-size: 15px;
            outline: none;
            transition: 0.2s;
        }

        input:focus, select:focus {
            border-color: #6366f1;
            box-shadow: 0 0 0 3px #6366f120;
        }

        .hint {
            font-size: 12px;
            color: #94a3b8;
        }

        .submit-btn {
            width: 100%;
            margin-top: 28px;
            padding: 16px;
            border: none;
            border-radius: 12px;
            color: white;
            background: linear-gradient(135deg, #4f46e5, #7c3aed);
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }

        .submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px #4f46e540;
        }

        .result {
            margin-top: 25px;
            padding: 23px;
            border-radius: 15px;
            background: #f1f5f9;
            border: 1px solid #e2e8f0;
            text-align: center;
        }

        .result h2 {
            font-size: 23px;
            margin-bottom: 10px;
        }

        .result p {
            line-height: 1.7;
            color: #475569;
        }

        .result.stay {
            background: #ecfdf5;
            border-color: #a7f3d0;
        }

        .result.leave {
            background: #fff1f2;
            border-color: #fecdd3;
        }

        .error {
            background: #fff7ed;
            color: #9a3412;
            border: 1px solid #fed7aa;
            border-radius: 10px;
            padding: 14px;
            margin-top: 20px;
            line-height: 1.6;
        }

        .footer {
            text-align: center;
            color: #94a3b8;
            font-size: 12px;
            margin-top: 25px;
            line-height: 1.7;
        }

        @media (max-width: 650px) {
            body {
                padding: 22px 12px;
            }

            .card {
                padding: 22px 17px;
            }

            .form-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>

<body>
<div class="container">

    <header class="header">
        <div class="logo">MACHINE LEARNING PROJECT</div>
        <h1>Employee Attrition Predictor</h1>
        <p class="subtitle">
            Enter employee details to get a prediction from your
            trained Gaussian Naive Bayes model.
        </p>
    </header>

    <main class="card">
        <h2 class="section-title">Employee Information</h2>
        <p class="section-description">
            Complete all eight fields and click the prediction button.
        </p>

        <form method="POST">
            <div class="form-grid">

                <div class="field">
                    <label for="education">Education</label>
                    <select name="education" id="education" required>
                        <option value="">Select education</option>
                        <option value="1">Bachelors</option>
                        <option value="2">Masters</option>
                        <option value="3">PHD</option>
                    </select>
                    <span class="hint">Encoded category</span>
                </div>

                <div class="field">
                    <label for="joining_year">Joining Year</label>
                    <input type="number" id="joining_year"
                           name="joining_year" min="2000" max="2035"
                           placeholder="e.g. 2022" required>
                </div>

                <div class="field">
                    <label for="city">City</label>
                    <select name="city" id="city" required>
                        <option value="">Select city</option>
                        <option value="1">Bangalore</option>
                        <option value="2">New Delhi</option>
                        <option value="3">Pune</option>
                    </select>
                </div>

                <div class="field">
                    <label for="payment_tier">Payment Tier</label>
                    <select name="payment_tier" id="payment_tier" required>
                        <option value="">Select tier</option>
                        <option value="1">Tier 1</option>
                        <option value="2">Tier 2</option>
                        <option value="3">Tier 3</option>
                    </select>
                </div>

                <div class="field">
                    <label for="age">Age</label>
                    <input type="number" id="age" name="age"
                           min="18" max="100" placeholder="e.g. 28" required>
                </div>

                <div class="field">
                    <label for="gender">Gender</label>
                    <select name="gender" id="gender" required>
                        <option value="">Select gender</option>
                        <option value="0">Female</option>
                        <option value="1">Male</option>
                    </select>
                </div>

                <div class="field">
                    <label for="ever_benched">Ever Benched?</label>
                    <select name="ever_benched" id="ever_benched" required>
                        <option value="">Select option</option>
                        <option value="0">No</option>
                        <option value="1">Yes</option>
                    </select>
                </div>

                <div class="field">
                    <label for="experience">Experience in Current Domain</label>
                    <input type="number" id="experience" name="experience"
                           min="0" max="60" placeholder="Years of experience"
                           required>
                </div>

            </div>

            <button class="submit-btn" type="submit">
                Predict Employee Attrition →
            </button>
        </form>

        {% if prediction is not none %}
            <div class="result {{ 'leave' if prediction == 1 else 'stay' }}">
                <h2>
                    {% if prediction == 1 %}
                        Attrition Predicted
                    {% else %}
                        Retention Predicted
                    {% endif %}
                </h2>
                <p>
                    {% if prediction == 1 %}
                        The model predicts that this employee may leave.
                    {% else %}
                        The model predicts that this employee may stay.
                    {% endif %}
                </p>
                <p><strong>Predicted class:</strong> {{ prediction }}</p>
            </div>
        {% endif %}

        {% if error %}
            <div class="error">{{ error }}</div>
        {% endif %}
    </main>

    <footer class="footer">
        Employee Attrition Prediction · Flask + Machine Learning<br>
        Predictions are estimates and should not be used as the sole basis
        for employment decisions.
    </footer>

</div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            education = int(request.form["education"])
            joining_year = int(request.form["joining_year"])
            city = int(request.form["city"])
            payment_tier = int(request.form["payment_tier"])
            age = int(request.form["age"])
            gender = int(request.form["gender"])
            ever_benched = int(request.form["ever_benched"])
            experience = int(request.form["experience"])

            # Keep the feature order consistent with model training.
            features = np.array([[
                education,
                joining_year,
                city,
                payment_tier,
                age,
                gender,
                ever_benched,
                experience
            ]])

            prediction = int(model.predict(features)[0])

        except (ValueError, KeyError, TypeError) as exc:
            error = "Please enter valid values in all fields."
        except Exception:
            error = (
                "Prediction failed. Check that the model expects "
                "these eight features in this exact order."
            )

    return render_template_string(
        HTML,
        prediction=prediction,
        error=error
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
