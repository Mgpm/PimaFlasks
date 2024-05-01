from flask import Flask,jsonify,request
import joblib
import numpy as np
import warnings
warnings.filterwarnings("ignore")
app = Flask(__name__)

model = joblib.load('model.pkl')

@app.route('/modelApi',methods=['GET'])
def index():
    cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age' ]
    query = []
    for i in  cols:
         query.append(request.args.get(i))
    if query:
        s={'Reponse':'0_ Vous êtres negatif au test diabètique,le modèle utilisé est la regression logistique avec un taux de reussite de 0.81'}
        pred = model.predict([np.array(query,dtype=float)])
        if pred[0]== 1:
            s={'Reponse':'1_ Vous êtes postif au test diabètique,le modèle utilisé est la regression logistique avec un taux de reussite de 0.81'}
        reponse = jsonify(s)
        return reponse



