from flask import Flask, render_template, request

import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from joblib import dump, load 

boston = load_boston()
X, y = boston.data, boston.target
df = pd.DataFrame(X, columns = boston.feature_names)[['CRIM', 'ZN', 'RM']]

X = df
lr = LinearRegression()
lr.fit(X, y)



app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/regression', methods = ['GET', 'POST'])
def reg():
    if request.method == 'POST':
        crim = float(request.form['crim'])
        zn = float(request.form['zn'])
        rm = float(request.form['rm'])
        test_point = np.array([crim, zn, rm]).reshape(1, -1)
        pred = lr.predict(test_point)
        return render_template('predictions.html', data = pred[0])
    return render_template('regression.html', data = df.head().to_html())

if __name__ == '__main__':
    app.run(debug = True)



