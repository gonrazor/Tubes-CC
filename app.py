from flask import Flask, render_template, request, jsonify
import pandas as pd
import h5py
import xgboost as xgb
import numpy as np

app = Flask(__name__)

# Load the XGBoost model
model = xgb.XGBRegressor()

# Provide the full path to the H5 file
h5_file_path = 'C:\\Users\\Lenovo\\Downloads\\tubes\\predicted_prices.h5'

with h5py.File(h5_file_path, 'r') as file:
    predicted_prices = file['predicted_prices'][:]

# Convert numpy array to DataFrame
predicted_prices_df = pd.DataFrame(predicted_prices, columns=['nama_tahun', 'nama_bulan', 'Prediksi_Harga'])

# Train the XGBoost model on the predicted prices
model.fit(predicted_prices_df[['nama_tahun', 'nama_bulan']], predicted_prices_df['Prediksi_Harga'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        month = int(request.form['month'])

        # Make predictions for the specified month
        new_data = {'nama_tahun': [2024], 'nama_bulan': [month]}
        new_df = pd.DataFrame(new_data)
        predicted_price = model.predict(new_df)

        # Return prediction and all predictions as JSON
        all_predictions = []
        for m in range(1, 13):
            new_data_all = {'nama_tahun': [2024], 'nama_bulan': [m]}
            new_df_all = pd.DataFrame(new_data_all)
            predicted_price_all = model.predict(new_df_all)
            all_predictions.append(float(predicted_price_all[0]))

        return jsonify({'predicted_price': float(predicted_price[0]), 'all_predictions': all_predictions})

if __name__ == '__main__':
    app.run(debug=True)
