import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained Naive Bayes model
with open('naive.pkl', 'rb') as f:
    model = pickle.load(f)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Employee Prediction Portal</title>
    <style>
        :root {
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --bg-gradient: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
            --card-bg: #ffffff;
            --text-color: #1f2937;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: var(--bg-gradient);
            color: var(--text-color);
            margin: 0;
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        .container {
            background: var(--card-bg);
            padding: 30px 40px;
            border-radius: 16px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
            width: 100%;
            max-width: 550px;
        }
        h2 {
            text-align: center;
            color: var(--primary);
            margin-bottom: 24px;
            font-size: 26px;
        }
        .form-group {
            margin-bottom: 16px;
        }
        label {
            display: block;
            margin-bottom: 6px;
            font-weight: 600;
            font-size: 14px;
        }
        input, select {
            width: 100%;
            padding: 10px 14px;
            border: 1px solid #d1d5db;
            border-radius: 8px;
            font-size: 14px;
            box-sizing: border-box;
            transition: border-color 0.2s;
        }
        input:focus, select:focus {
            border-color: var(--primary);
            outline: none;
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
        }
        button {
            width: 100%;
            background: var(--primary);
            color: white;
            border: none;
            padding: 12px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            margin-top: 10px;
            transition: background 0.2s;
        }
        button:hover {
            background: var(--primary-hover);
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            background: #e0e7ff;
            color: #3730a3;
            border-radius: 8px;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Employee Prediction System</h2>
        <form method="POST" action="/predict">
            <div class="form-group">
                <label>Education:</label>
                <select name="Education" required>
                    <option value="0">Bachelors</option>
                    <option value="1">Masters</option>
                    <option value="2">PHD</option>
                </select>
            </div>
            <div class="form-group">
                <label>Joining Year:</label>
                <input type="number" name="JoiningYear" placeholder="e.g., 2018" required>
            </div>
            <div class="form-group">
                <label>City:</label>
                <select name="City" required>
                    <option value="0">Bangalore</option>
                    <option value="1">Pune</option>
                    <option value="2">New Delhi</option>
                </select>
            </div>
            <div class="form-group">
                <label>Payment Tier:</label>
                <select name="PaymentTier" required>
                    <option value="1">Tier 1</option>
                    <option value="2">Tier 2</option>
                    <option value="3">Tier 3</option>
                </select>
            </div>
            <div class="form-group">
                <label>Age:</label>
                <input type="number" name="Age" placeholder="e.g., 28" required>
            </div>
            <div class="form-group">
                <label>Gender:</label>
                <select name="Gender" required>
                    <option value="0">Male</option>
                    <option value="1">Female</option>
                </select>
            </div>
            <div class="form-group">
                <label>Ever Benched:</label>
                <select name="EverBenched" required>
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                </select>
            </div>
            <div class="form-group">
                <label>Experience In Current Domain:</label>
                <input type="number" name="ExperienceInCurrentDomain" placeholder="e.g., 3" required>
            </div>
            <button type="submit">Predict</button>
        </form>
        {% if prediction is not none %}
        <div class="result">
            Prediction Result: {{ prediction }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE, prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['Education']),
            float(request.form['JoiningYear']),
            float(request.form['City']),
            float(request.form['PaymentTier']),
            float(request.form['Age']),
            float(request.form['Gender']),
            float(request.form['EverBenched']),
            float(request.form['ExperienceInCurrentDomain'])
        ]
        
        final_features = np.array([features])
        prediction = model.predict(final_features)[0]
        
        result_text = f"Will Leave / Positive Class ({prediction})" if prediction == 1 else f"Will Stay / Negative Class ({prediction})"
        
        return render_template_string(HTML_TEMPLATE, prediction=result_text)
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
