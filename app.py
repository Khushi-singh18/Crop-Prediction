from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open('crop_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the label encoder
with open('label_encoder.pkl', 'rb') as encoder_file:
    le = pickle.load(encoder_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    input_data = np.array([data])
    prediction = model.predict(input_data)
    crop_name = le.inverse_transform(prediction)[0]
    return render_template('index.html', prediction_text=f"🌾 Recommended Crop: {crop_name}")

if __name__ == '__main__':
    app.run(debug=True)
