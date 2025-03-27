from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("groundwater_model.pkl")

@app.route("/")
def home():
    return "Welcome to the Groundwater Level Prediction API!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        year = int(data["year"])
        station = int(data["station"])

        input_data = np.array([[year]])
        prediction = model.predict(input_data)

        return jsonify({"predicted_wse": float(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
