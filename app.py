import pickle
from flask import Flask,request,app,jsonify,url_for,render_template # type: ignore
import numpy as np # type: ignore
import pandas as np # type: ignore

app = Flask(__name__)# starting point of app running

# load the model
model = pickle.load(open('reg_model.pkl','rb'))

@app.route('/') # 1st route return url
def home():
    return render_template('home.html') # return html page as render_template

# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     data = request.json['data']
#     print(data)
#     print(np.array(list(data.values))).reshape(1,-1)
#     new_data = scaler.transform(np.array(list(data.values)).reshape(1,-1)) # type: ignore

#     output =model.predict(new_data)
#     print(output[0])
#     return jsonify(output[0])

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']  # Getting the 'data' part from the request
    data_array = list(data.values())  # Convert dict values to list
    final_input = np.array(data_array).reshape(1, -1)  # Make it a 2D array for model
    output = model.predict(final_input)[0]  # Predict and get first value
    return jsonify(output[0])  # Return as JSON



if __name__=='__main__':
    app.run(debug=True)
