import pickle
from flask import Flask,request,app,jsonify,url_for,render_template

import numpy as np
import pandas as pd

app=Flask(__name__)
#loading the model
model=pickle.load(open("svm_model.pkl","rb"))
scaler=pickle.load(open("scaling.pkl","rb"))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=["POST"])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output=model.predict(new_data)
    print(output[0])
    return jsonify(int(output[0]))

@app.route('/about')
def about():
    about_info = {
        'name': 'Nishchala Mukku',
        'email': 'mnishchala@gmail.com',
        'github': 'https://github.com/Nishchalam',
        'linkedin': 'https://www.linkedin.com/in/nishchala-mukku/',
        'resume' : 'https://drive.google.com/file/d/1YSt72SX_3jlal10DKDek-KTq7P1JjLC_/view'
    }
    return render_template('about.html', about_info=about_info)


if __name__ == "__main__":
    app.run(debug=True)

