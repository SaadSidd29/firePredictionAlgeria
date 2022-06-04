import pickle
from flask import Flask,request,app,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

app=Flask(__name__)
model=pickle.load(open('model_predict.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)

    new_data=[list(data.values())]
    output=model.predict(new_data)
    return jsonify(output)

@app.route('/predict',methods=['POST'])
def predict():
    
    print("Form Values : ",request.form.values())
    data=[float(x) for x in request.form.values()]
    final_features = [np.array(data)]

    sc=StandardScaler()
    final_features_scaled=sc.fit_transform(final_features)

    print("Data is : ",data)
    print("Scaled Data is : ",final_features_scaled)
 
    output=model.predict(final_features_scaled)[0]
    print("Result : ",output)
    return render_template('home.html', prediction_text="There will be {}".format(output))


if __name__=="__main__":
    app.run(debug=True)