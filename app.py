from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    species = request.form['species']
    length1 = float(request.form['length1'])
    length2 = float(request.form['length2'])
    length3 = float(request.form['length3'])
    height = float(request.form['height'])
    width = float(request.form['width'])

    species_map = {
        'Bream': 0, 'Roach': 1, 'Perch': 2, 'Pike': 3,
        'Smelt': 4, 'Parkki': 5, 'Whitefish': 6
    }
    species_encoded = [0] * 6
    species_encoded[species_map[species] - 1] = 1

    features = [length1, length2, length3, height, width] + species_encoded
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)

