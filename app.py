import pickle
from flask import Flask, render_template, request
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('naive.pkl', 'rb'))


@app.route('/')
def home():
  return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
  try:
    # Extract input values from the form
    education = request.form['Education']
    joining_year = int(request.form['JoiningYear'])
    city = request.form['City']
    payment_tier = int(request.form['PaymentTier'])
    age = int(request.form['Age'])
    gender = request.form['Gender']
    ever_benched = request.form['EverBenched']
    experience = int(request.form['ExperienceInCurrentDomain'])

    # Map categorical features (adjust mappings based on your training encoding)
    edu_map = {'Bachelors': 0, 'Masters': 1, 'PHD': 2}
    city_map = {'Bangalore': 0, 'New Delhi': 1, 'Pune': 2}
    gender_map = {'Male': 0, 'Female': 1}
    benched_map = {'No': 0, 'Yes': 1}

    input_data = pd.DataFrame([[
        edu_map.get(education, 0),
        joining_year,
        city_map.get(city, 0),
        payment_tier,
        age,
        gender_map.get(gender, 0),
        benched_map.get(ever_benched, 0),
        experience,
    ]], columns=[
        'Education',
        'JoiningYear',
        'City',
        'PaymentTier',
        'Age',
        'Gender',
        'EverBenched',
        'ExperienceInCurrentDomain',
    ])

    # Make prediction
    prediction = model.predict(input_data)[0]
    result_text = (
        'Likely to Leave the Organization'
        if prediction == 1
        else 'Likely to Stay'
    )

    return render_template(
        'index.html', prediction_text=f'Prediction: {result_text}'
    )
  except Exception as e:
    return render_template('index.html', prediction_text=f'Error: {str(e)}')


if __name__ == '__main__':
  app.run(debug=True)
