import numpy as np
from flask import Flask,render_template,request
import pickle

app = Flask(__name__)
classifiermodel = pickle.load(open('diabeticpredmodel.pickle', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    values = [np.array(int_features)]
    prediction = classifiermodel.predict(values)
    return render_template('index.html',prediction_text='output value is: {}'.format(prediction))
	
		
if __name__ == "__main__":
    app.run(debug=True)