from flask import Flask, request, jsonify
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import os

app = Flask(__name__)

# Load and preprocess data
dataset = pd.read_csv("Dataset/CropDataset.csv", usecols=['variety', 'max_price', 'Rainfall'])
dataset.fillna(0, inplace=True)

# Encode variety (categorical) column
le = LabelEncoder()
dataset['variety'] = le.fit_transform(dataset['variety'].astype(str))

# Features and target
X = dataset[['variety', 'Rainfall']]
Y = dataset[['max_price']]

# Scale data
sc_X = MinMaxScaler()
sc_Y = MinMaxScaler()

X_scaled = sc_X.fit_transform(X)
Y_scaled = sc_Y.fit_transform(Y)

# Train model
model = RandomForestRegressor()
model.fit(X_scaled, Y_scaled.ravel())

# Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received data:", data)

        variety = data['variety']
        rainfall = data['rainfall']

        # Ensure variety is valid
        if variety not in le.classes_:
            return jsonify({"error": "Invalid variety"}), 400

        # Encode and scale input
        variety_encoded = le.transform([variety])[0]
        input_features = [[variety_encoded, rainfall]]
        input_scaled = sc_X.transform(input_features)

        # Predict and inverse scale
        prediction_scaled = model.predict(input_scaled)
        prediction = sc_Y.inverse_transform([[prediction_scaled[0]]])[0][0]

        print("Prediction:", prediction)

        return jsonify({"prediction": round(float(prediction), 2)})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": "Cannot process your request", "details": str(e)}), 500

# Run the app on the correct host/port for Render
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
