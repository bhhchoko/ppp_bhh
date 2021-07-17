import pandas as pd
from flask import Flask, jsonify, request, render_template
import pickle
import numpy as np
model = pickle.load(open('D_tree_Model.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    data5 = request.form['e']
    data6 = request.form['f']
    data7 = request.form['g']
    data8 = request.form['h']
    data9 = request.form['i']
    data10 = request.form['j']
    data11 = request.form['k']
    data12 = request.form['l']
    data13 = request.form['m']
    data14 = request.form['n']
    data15 = request.form['o']
    data16 = request.form['p']
    data17 = request.form['q']
    data18 = request.form['r']
    data19 = request.form['s']
    data20 = request.form['t']
    data21 = request.form['u']
    data22 = request.form['v']
    data23 = request.form['w']
    data24 = request.form['x']
    data25 = request.form['y']
    data26 = request.form['z']
    arr = np.array([[data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, data14, data15, data16, data17, data18, data19, data20, data21, data22, data23, data24, data25, data26 ]]) 
    pred = model.predict(arr)
    return render_template('resultat.html', data = pred)

if __name__ == '__main__':
    app.run(port = 5000, debug=True)

    




