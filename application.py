import learn as learn
from flask import Flask, render_template, request
import pandas as pd
import pickle
import sklearn.linear_model
import numpy as np




app = Flask(__name__)
model = pickle.load(open("LinearRegressionModel.pkl", 'rb'))
car = pd.read_csv('Cleaned Car.csv')


@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    companies.insert(0,"Select Company")
    fuel_type = car['fuel_type'].unique()
    return render_template('index.html', companies=companies, car_models=car_models, years=year, fuel_type=fuel_type)

    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    car_models = request.form.get('car_models')
    years = int(request.form.get('years'))

    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kilo_driven'))


    prediction = model.predict(pd.DataFrame([[car_models, company, years, kms_driven, fuel_type]],
                                            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))


    return str(np.round(prediction[0],2))


if __name__ == "__main__":
    app.run(debug=True)
