# %%
import flask as fl
import pandas as pd
import numpy as np
import pickle
import joblib
from joblib import dump, load

# %%
#create instance of flask class
from flask import Flask, render_template, request
app = fl.Flask(__name__, template_folder='templates')

# %%
@app.route('/')
def home():
    return render_template('home.html')

# %%
#@app.route('/', methods=['GET','POST'])
@app.route('/predict', methods=['GET','POST'])

def predict():
    #if fl.request.method == 'GET':
        #return (fl.render_template('main.html'))
    # retrieve data from formular            
    if fl.request.method == 'POST':
        
        print(fl.request.form.get('age'))
        print(fl.request.form.get('sex'))
        print(fl.request.form.get('cp'))
        print(fl.request.form.get('trtbps'))
        print(fl.request.form.get('chol'))
        print(fl.request.form.get('fbs'))
        print(fl.request.form.get('restecg'))
        print(fl.request.form.get('thalachh'))
        print(fl.request.form.get('exng'))
        print(fl.request.form.get('oldpeak'))
        print(fl.request.form.get('slp'))
        print(fl.request.form.get('caa'))
        print(fl.request.form.get('thall'))
        try:
              age=float(fl.request.form['age'])
              sex=float(fl.request.form['sex'])
              cp=float(fl.request.form['cp'])
              trtbps=float(fl.request.form['trtbps'])
              chol=float(fl.request.form['chol'])
              fbs=float(fl.request.form['fbs'])
              restecg=float(fl.request.form['restecg'])
              thalachh=float(fl.request.form['thalachh'])
              exng=float(fl.request.form['exng'])
              oldpeak=float(fl.request.form['oldpeak'])
              slp=float(fl.request.form['slp'])
              caa=float(fl.request.form['caa'])
              thall=float(fl.request.form['thall'])
              
              pred_args=[age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall]
              pred_arr=np.array(pred_args)
              preds=pred_arr.reshape(1,-1)
              
              filename='RandomForest_model_ts_20.pkl'
              model = open(filename, 'rb')
              rf_model=joblib.load(model)
              model_prediction=rf_model.predict(preds)
              
        except ValueError:
              return "Please Enter valid values"

    return render_template('predict.html',prediction=model_prediction)

# %%
if __name__=='__main__':
    app.run()