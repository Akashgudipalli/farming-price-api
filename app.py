from flask import Flask, request, jsonify
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

app = Flask(__name__)

# Load and preprocess data
dataset = pd.read_csv("Dataset/CropDataset.csv", usecols=['variety', 'max_price', 'Rainfall'])
dataset.fillna(0, inplace=True)

le = LabelEncoder()
dataset['variety'] = le.fit_transform(dataset['variety'].astype(str))

X = dataset[['variety', 'Rainfall']]
Y = dataset[['max_price']]

sc_X = MinMaxScaler()
sc_Y = MinMaxScaler()

X_scaled = sc_X.fit_transform(X)
Y_scaled = sc_Y.fit_transform(Y)

model = RandomForestRegressor()
model.fit(X_scaled, Y_scaled.ravel())

# API route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received data:", data)

        variety = data['variety']
        rainfall = data['rainfall']

        if variety not in le.classes_:
            return jsonify({"error": "Invalid variety"}), 400

        variety_encoded = le.transform([variety])[0]
        features = [[variety_encoded, rainfall]]

        prediction = model.predict(features)[0]
        print("Prediction:", prediction)

        return jsonify({"prediction": float(prediction)})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": "Cannot process your request", "details": str(e)}), 500


import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
