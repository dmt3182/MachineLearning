import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from tensorflow.python import keras


data = pd.read_csv(r'E:\1.DEEPAK Data Science\GITREPO\Deep Learning\Diabetesprediction\diabetes.csv')
data.info()
X = data.iloc[:,0:7]
Y  = data.iloc[:,-1]

sc = StandardScaler()
X = sc.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state= 20)
# Creating the model

model = Sequential([
                    Dense(128,activation='relu',kernel_initializer= 'glorot_uniform'),
                    Dropout(0.5),
                    Dense(64,activation='relu',kernel_initializer= 'glorot_uniform'),
                    Dropout(0.2),
                    Dense(32,activation='relu',kernel_initializer= 'glorot_uniform'),
                    Dropout(0.2),
                    Dense(1, activation= 'sigmoid')])

# compiling the modeland define optimizer and loss function
model.compile(optimizer = 'SGD',loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X_train, Y_train,batch_size=32, epochs= 200,validation_split= 0.2 )   

model.save('diabetes.hdf5')

# Y_pred = model.predict(X_test)
# Y_pred = Y_pred>0.5
# acc_score = accuracy_score(Y_pred,Y_test)
# print (acc_score)

