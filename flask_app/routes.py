#importing all the necessary libraries
from flask import render_template, request
import pickle

import warnings 
warnings.filterwarnings(action= 'ignore')

from flask_app import app
from .helper import query_point_creator

# Load the model
model = pickle.load(open('flask_app\models\model.pkl','rb'))

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == "POST":
        
        # Get values through input bars
        q1 = request.form.get("1stQuestion")
        q2 = request.form.get("2ndQuestion")

        query = query_point_creator(q1, q2)

        # Get prediction
        prediction = model.predict(query)[0]
        
        if prediction:
            result = "Dublicate"
        else:
            result = "Non-Dublicate"
    else:
        result = ""

    return render_template("index.html", output = result)