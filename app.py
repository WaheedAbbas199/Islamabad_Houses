
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import pickle
from flask import Flask, render_template

app = Flask(__name__)
data = pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/pridicted')
def pridicted():
    locations = sorted(data['location'].unique())
    return render_template('pridicted.html', locations=locations)

@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/blogs')
def blogs():
    return render_template('blogs.html')
@app.route('/Converter')
def Converter():
    return render_template('Converter.html')
@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bedrooms = request.form.get('bedrooms')
    baths = request.form.get('baths')
    Total_Area = request.form.get('Total_Area')

    # Validate and convert inputs
    try:
        bedrooms = int(bedrooms)
        baths = int(baths)
        Total_Area = float(Total_Area)
    except ValueError:
        return "Invalid input: Please ensure all fields are filled out correctly."

    print(location, bedrooms, baths, Total_Area)
    input_data = pd.DataFrame([[location, bedrooms, baths, Total_Area]], columns=["location", "bedrooms", "baths", "Total_Area"])
    prediction = pipe.predict(input_data)[0] * 1e5

    return str(np.around(prediction,2))

if __name__ == '__main__':
    app.run(debug=True, port=5002)

