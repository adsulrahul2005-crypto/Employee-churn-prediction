import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained Naive Bayes model[cite: 1]
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
            --bg-gradient: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: var(--bg-gradient);
            color: #1e293b;
            margin: 0;
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        
        /* Interactive Model Frame Styles */
        .interactive-frame-container {
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(255, 255, 255, 0.8);
            border-radius: 20px;
            box-shadow: 0 20px 35px -10px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            backdrop-filter: blur(10px);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
            width: 100%;
            max-width: 550px;
            padding: 35px 40px;
            box-sizing: border-box;
        }

        .interactive-frame-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #4f46e5, #06b6d4, #10b981);
        }

        .model-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding-bottom: 16px;
            border-bottom: 1px solid #e2e8f0;
            margin-bottom: 24px;
        }

        .model-header h2 {
            margin: 0;
            color: var(--primary);
            font-size: 24px;
        }

        .model-badge {
            background: #e0e7ff;
            color: #4f46e5;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }

        .model-badge::before {
            content: '';
            width: 6px;
            height: 6px;
            background: #4f46e5;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(0.95); opacity: 0.8; }
            50% { transform: scale(1.2); opacity: 1; box-shadow: 0 0 0 4px rgba(79, 70, 229, 0.2); }
            100% { transform: scale(0.95); opacity: 0.8; }
        }

        .interactive-input-group {
            position: relative;
            margin-bottom: 18px;
        }

        .interactive-input-group label {
            font-size: 13px;
            font-weight: 600;
            color: #475569;
            margin-bottom: 6px;
            display: block;
        }

        .interactive-input-group input,
        .interactive-input-group select {
            width: 100%;
            padding: 11px 14px;
            background: #ffffff;
            border: 2px solid #e2e8f0;
            border-radius: 10px;
            font-size: 14px;
            color: #1e293b;
            transition: all 0.25s ease;
            box-sizing: border-box;
        }

        .interactive-input-group input:focus,
        .interactive-input-group select:focus {
            border-color: #4f46e5;
            outline: none;
            box-shadow: 0 0 0 4px rgba(79, 70, 229, 0.12);
        }

        .interactive-submit-btn {
            background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
            color: white;
            border: none;
            width: 100%;
            padding: 13px;
            border-radius: 10px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
            transition: all 0.25s ease;
            margin-top: 5px;
        }

        .interactive-submit-btn:hover {
            background: linear-gradient(135deg, #4338ca 0%, #3730a3 100%);
            box-shadow: 0 6px 16px rgba(79, 70, 229, 0.4);
            transform: translateY(-1px);
        }

        .prediction-output-card {
            margin-top: 22px;
            padding: 16px;
            border-radius: 12px;
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
            color: #166534;
            text-align: center;
            font-weight: 600;
            font-size: 16px;
            animation: slideUp 0.3s ease-out forwards;
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="interactive-frame-container">
        <div class="model-header">
            <h2>Employee Predictor</h2>
            <span class="model-badge">Active Model</span>
        </div>
        
        <form method="POST" action="/predict">
            <div class="interactive-input-group">
                <label>Education</label>
                <select name="Education" required>
                    <option value="0">Bachelors</option>
                    <option value="1">Masters</option>
                    <option value="2">PHD</option>
                </select>
            </div>
            <div class="interactive-input-group">
                <label>Joining Year</label>
                <input type="number" name="JoiningYear" placeholder="e.g., 2018" required>
            </div>
            <div class="interactive-input-group">
                <label>City</label>
                <select name="City" required>
                    <option value="0">Bangalore</option>
                    <option value="1">Pune</option>
                    <option value="2">New Delhi</option>
                </select>
            </div>
            <div class="interactive-input-group">
                <label>Payment Tier</label>
                <select name="PaymentTier" required>
                    <option value="1">Tier 1</option>
                    <option value="2">Tier 2</option>
                    <option value="3">Tier 3</option>
                </select>
            </div>
            <div class="interactive-input-group">
                <label>Age</label>
                <input type="number" name="Age" placeholder="e.g., 28" required>
            </div>
            <div class="interactive-input-group">
                <label>Gender</label>
                <select name="Gender" required>
                    <option value="0">Male</option>
                    <option value="1">Female</option>
                </select>
            </div>
            <div class="interactive-input-group">
                <label>Ever Benched</label>
                <select name="EverBenched" required>
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                </select>
            </div>
            <div class="interactive-input-group">
                <label>Experience In Current Domain</label>
                <input type="number" name="ExperienceInCurrentDomain" placeholder="e.g., 3" required>
            </div>
            
            <button type="submit" class="interactive-submit-btn">Run Prediction</button>
        </form>

        {% if prediction is not none %}
        <div class="prediction-output-card">
            Result: {{ prediction }}
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
