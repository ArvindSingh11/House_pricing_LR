# import pickle
# from flask import Flask,request,app,jsonify,url_for,render_template # type: ignore
# import numpy as np # type: ignore
# import pandas as pd # type: ignore

# app = Flask(__name__)# starting point of app running

# # load the model
# model = pickle.load(open('reg_model.pkl','rb'))

# @app.route('/') # 1st route return url
# def home():
#     return render_template('home.html') # return html page as render_template

# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     data = request.json['data']
#     print(data)
#     print(np.array(list(data.values))).reshape(1,-1)
#     new_data = scalar.transform(np.array(list(data.values)).reshape(1,-1)) # type: ignore

#     output =model.predict(new_data)
#     print(output[0])
#     return jsonify(output)

# # @app.route('/predict', methods=['POST'])
# # def predict_api():
# #     data = request.json['data']  # Getting the 'data' part from the request
# #     data_array = list(data.values())  # Convert dict values to list
# #     scalar = pickle.load(open('scaler.pkl', 'rb'))
# #     final_input = np.array(data_array).reshape(1, -1)  # Make it a 2D array for model
# #     output = model.predict(final_input)[0]  # Predict and get first value
# #     return jsonify({'prediction': output})  # Return as JSON properly




# if __name__=='__main__':
#     app.run(debug=True)

import pickle
from flask import Flask, request, app, jsonify, url_for, render_template  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # <-- corrected here

app = Flask(__name__)

# Load the model
model = pickle.load(open('reg_model.pkl', 'rb'))

# (Optional) If you have a scaler
# scalar = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']  # Get data from request
    data_array = list(data.values())
    final_input = np.array(data_array).reshape(1, -1)

    # final_input = scaler.transform(final_input)

    output = model.predict(final_input)[0]  # Predict
    return jsonify({'prediction': output})  # Return JSON response

if __name__ == '__main__':
    app.run(debug=True)
