import numpy as np
from flask import Flask, render_template, request
from joblib import load

app = Flask(__name__)

model = load('reg_model.joblib')
scaler = load('scaler.pkl')

@app.route('/')
def display_form():
    return render_template('predict.html')

@app.route("/predict", methods=['POST'])
def predict_price():
    if request.method == "POST":
        sqft_lot = int(request.form['sqft_lot'])
        floors = float(request.form['floors'])
        bedrooms = int(request.form['bedrooms'])
        bathrooms = float(request.form['bathrooms'])

        input_vector = np.array([[sqft_lot, floors, bedrooms, bathrooms]])
        input_scaled = scaler.transform(input_vector)
        predicted_price = model.predict(input_scaled)

        return render_template('predict.html', price = round(predicted_price[0], 2))

if __name__ == '__main__':
    app.run()