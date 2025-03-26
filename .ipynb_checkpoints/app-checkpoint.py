from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        diagonal = float(request.form['diagonal'])
        height_left = float(request.form['height_left'])
        height_right = float(request.form['height_right'])
        margin_low = float(request.form['margin_low'])
        margin_up = float(request.form['margin_up'])
        length = float(request.form['length'])

        # Combine inputs into a NumPy array
        input_data = np.array([[diagonal, height_left, height_right, margin_low, margin_up, length]])

        # Predict
        prediction = model.predict(input_data)[0]

        # Interpret result
        result = "Original Currency" if prediction == 1 else "Fake Currency"

        return render_template("index.html", prediction_text=f"The bill is: {result}")

    except Exception as e:
        print("Prediction error:", e)
        return render_template("index.html", prediction_text="Error: Invalid input!")


if __name__ == "__main__":
    app.run(debug=True)
