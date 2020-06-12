# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 15:58:05 2020

@author: Rahul3.Tiwari
"""

# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'model-salary.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        exp = [float(x) for x in request.form.values()]
        
        
        data = np.array([exp])
        my_prediction = model.predict(data)
        
        return render_template('index.html', prediction_text=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)