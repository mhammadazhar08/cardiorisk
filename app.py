import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the full pipeline (imputer + scaler + model). imbalanced-learn must be installed.
with open(os.path.join(BASE_DIR, 'chd_pipeline.pkl'), 'rb') as f:
    model = pickle.load(f)

# Same columns, same order as training (education dropped)
FEATURES = ['male', 'age', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'prevalentStroke',
            'prevalentHyp', 'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI',
            'heartRate', 'glucose']

NUMERIC = ['age', 'cigsPerDay', 'totChol', 'sysBP',
           'diaBP', 'BMI', 'heartRate', 'glucose']
REQUIRED = {'age': 'Age', 'sysBP': 'Systolic blood pressure',
            'diaBP': 'Diastolic blood pressure'}


def yes(value):
    return 1 if str(value).strip().lower() == "yes" else 0


def to_number(value):
    """Convert a form value to float; blank -> NaN so the pipeline's imputer fills it."""
    value = (value or "").strip()
    return float(value) if value else np.nan


def predict(model, male, age, currentSmoker, cigsPerDay, BPMeds, prevalentStroke,
            prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose):
    patient = pd.DataFrame([{
        'male': 1 if str(male).strip().lower() == "male" else 0,
        'age': age,
        'currentSmoker': yes(currentSmoker),
        'cigsPerDay': cigsPerDay,
        'BPMeds': yes(BPMeds),
        'prevalentStroke': yes(prevalentStroke),
        'prevalentHyp': yes(prevalentHyp),
        'diabetes': yes(diabetes),
        'totChol': totChol,
        'sysBP': sysBP,
        'diaBP': diaBP,
        'BMI': BMI,
        'heartRate': heartRate,
        'glucose': glucose,
    }])[FEATURES]

    prediction = int(model.predict(patient)[0])
    probability = float(model.predict_proba(patient)[0, 1]) if hasattr(
        model, 'predict_proba') else None
    return prediction, probability


def notable_readings(form, n):
    """Readings outside common clinical reference ranges (general guidance, not model output)."""
    factors = []
    if n['sysBP'] >= 140 or n['diaBP'] >= 90:
        factors.append(
            ("High blood pressure", f"{n['sysBP']:.0f}/{n['diaBP']:.0f} mmHg"))
    if n['totChol'] >= 240:
        factors.append(("High total cholesterol", f"{n['totChol']:.0f} mg/dL"))
    if n['glucose'] >= 126:
        factors.append(("High glucose", f"{n['glucose']:.0f} mg/dL"))
    if n['BMI'] >= 30:
        factors.append(("BMI in obese range", f"{n['BMI']:.1f} kg/m²"))
    if yes(form.get('currentSmoker')):
        cigs = n['cigsPerDay']
        factors.append(
            ("Current smoker", f"{cigs:.0f} per day" if not np.isnan(cigs) else "Yes"))
    if yes(form.get('diabetes')):
        factors.append(("Diabetes", "Yes"))
    if yes(form.get('prevalentStroke')):
        factors.append(("Previous stroke", "Yes"))
    return factors


# Routes
@app.route("/")
def index():
    return render_template('index.html', form={}, result=None)


@app.route('/predict', methods=['POST'])
def predict_route():
    form = request.form
    try:
        numbers = {name: to_number(form.get(name)) for name in NUMERIC}
    except ValueError:
        return render_template('index.html', form=form,
                               error="Please enter valid numbers in the measurement fields.")

    missing = [label for name, label in REQUIRED.items()
               if np.isnan(numbers[name])]
    if missing:
        return render_template('index.html', form=form,
                               error="Please fill in: " + ", ".join(missing) + ".")

    prediction, probability = predict(
        model,
        male=form.get('male', ''),
        currentSmoker=form.get('currentSmoker', ''),
        BPMeds=form.get('BPMeds', ''),
        prevalentStroke=form.get('prevalentStroke', ''),
        prevalentHyp=form.get('prevalentHyp', ''),
        diabetes=form.get('diabetes', ''),
        **numbers,
    )

    result = {
        'high_risk': prediction == 1,
        'probability': probability,
        'factors': notable_readings(form, numbers),
    }
    return render_template('index.html', form=form, result=result)


if __name__ == '__main__':
    app.run(debug=True)
