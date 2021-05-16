import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow as tf

app = Flask(__name__)
# Recreate the exact same model, including its weights and the optimizer
model = tf.keras.models.load_model('model.hdf5')


@app.route('/')

def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    print (int_features) 
        
    # final_features = [np.array(int_features)]
    # prediction = model.predict(final_features)

    # output = round(prediction[0], 2)

    # return render_template('index.html', prediction_text= output)


if __name__ == "__main__":
    app.run(debug=True)