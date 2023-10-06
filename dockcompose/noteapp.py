from flask import Flask, render_template, request
from  pathlib import Path
import numpy as np
import pandas as pd
import pickle



app = Flask(__name__)

scl= Path(__file__).absolute().parent
fl= scl / "rfcmodel.pkl"
rfc = pickle.load(open(fl, 'rb'))


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods= ['POST'])
def note_authentication():
    variance=request.form.get("variance")
    skewness=request.form.get("skewness")
    curtosis=request.form.get("curtosis")
    entropy=request.form.get("entropy")
    
    result=rfc.predict([[variance,skewness,curtosis,entropy]])
    
    return render_template('index.html', result ="Banknote class :" + str(result))


@app.route('/predict_file', methods=['POST'])
def file_pred():
    file_name=request.files["file"]
    note_df= pd.read_csv(file_name)
    note_df.dropna(inplace=True)
    result=rfc.predict(note_df)
    return render_template('index.html', result ="Banknote class :" + str(list(result)))


if __name__== '__main__':
    app.run(debug=True, port=5000)