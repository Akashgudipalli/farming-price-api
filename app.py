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
    data = request.get_json()
    variety = data['variety']
    rainfall = float(data['rainfall'])

    try:
        variety_encoded = le.transform([variety])[0]
    except:
        return jsonify({'error': 'Invalid variety'}), 400

    input_data = sc_X.transform([[variety_encoded, rainfall]])
    prediction_scaled = model.predict(input_data)
    prediction = sc_Y.inverse_transform([[prediction_scaled[0]]])[0][0]

    return jsonify({'prediction': round(prediction, 2)})

if __name__ == '__main__':
    app.run(debug=True)