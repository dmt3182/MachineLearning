import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('diabetes.hdf5')

@app.route('/')
def home():
    return render_template('diabetes1.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    # '''
    int_features =  request.form.values()
    final_features = np.array(int_features)
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('diabetes1.html', prediction_text= output)


if __name__ == "__main__":
    app.run(debug=True)