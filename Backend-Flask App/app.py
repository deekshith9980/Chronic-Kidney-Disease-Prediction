import requests
import pandas as pd
import numpy as np

from flask import Flask, request, render_template, url_for, redirect
import pickle
import requests
from main import lgpredict

app = Flask(__name__)

model = pickle.load(open('C:/Users/DEEKSHITH/Downloads/mini_project_akki/mpcopy/Chronic-Kidney-Disease-Detection-Using-Machine-Learning-main/CKD.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html') 
    
@app.route('/Prediction', methods = ['POST','GET'])
def prediction():  
    return render_template('indexnew.html')
@app.route('/Home', methods = ['POST','GET'])
def my_home():
    return render_template('home.html')

@app.route('/predict', methods = ['POST'])  
def predict():
    
    blood_urea = request.form["blood_urea"]
    blood_glucose_random = request.form["blood glucose random"]
    coronary_artery_disease = request.form["coronary_artery_disease"]
    anemia = request.form["anemia"]
    pus_cell = request.form["pus_cell"]
    red_blood_cells = request.form["red_blood_cells"]
    diabetesmellitus = request.form["diabetesmellitus"]
    pedal_edema = request.form["pedal_edema"]

    if (coronary_artery_disease == 'Yes'):
        c1 = 1
    else:
        c1 = 0
    if (anemia == 'Yes'):
        a1 = 1
    else:
        a1 = 0
    if (pus_cell == 'Normal'):
        p1 = 1
    else:
        p1 = 0
    if (red_blood_cells == 'Normal'):
        r1 = 1
    else:
        r1 = 0
    if(diabetesmellitus == 'Yes'):
        d1 = 1
    else:
        d1 = 0
    if(pedal_edema == 'Yes'):
        p2 = 1
    else:
        p2 = 0
    
    t = [[int(blood_urea),int(blood_glucose_random),int(c1),int(a1),int(p1),int(r1),int(d1),int(p2)]]
    print(t)

    pred=lgpredict(int(blood_urea),int(blood_glucose_random),int(c1),int(a1),int(p1),int(r1),int(d1),int(p2))

    print(pred)

    
     
    return render_template('result.html',pred = pred)

if __name__ == '__main__':
    app.run(debug = True)   
