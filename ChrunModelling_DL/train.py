# input layer 
# hidden layer
# number of neurons in each hidden layer
# output layer 
# weights initialization(uniform, gorat uniform, gorat normal, he normal, he uniform)
# optimizer
# activation function (sigmoid,relu)
# drop out

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

data = pd.read_csv(r'E:\1.DEEPAK Data Science\GITREPO\Deep Learning\ChrunModelling_DL\Churn_Modelling.csv')
data.info()
X = data.iloc[:,3:13]
Y = data.iloc[:,13]
geography = pd.get_dummies(X['Geography'])
gender = pd.get_dummies(X['Gender'])
X = pd.concat([X,gender,geography],axis = 1)
X.drop(columns= ['Geography', 'Gender'], axis = 1, inplace = True)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state =101 , test_size = 0.2)

# feature scaling
sc = StandardScaler()
X = sc.fit_transform(X)

# Create DL model

model = Sequential([
                    Dense(128,activation='relu',kernel_initializer='he_uniform'),
                    Dropout(0.5),
                    Dense(64,activation='relu',kernel_initializer='he_uniform'),
                    Dropout(0.6),
                    Dense(32,activation='relu',kernel_initializer='he_uniform'),
                    Dropout(0.4),
                    Dense(16,activation='relu',kernel_initializer='he_uniform'),
                    Dropout(0.2),
                    Dense(8,activation='relu',kernel_initializer='he_uniform'),
                    Dropout(0.4),
                    Dense(4,activation='relu',kernel_initializer='he_uniform'),
                    Dropout(0.2),
                    Dense(1,activation = 'sigmoid')])

# compiling the modeland define optimizer and loss function
model.compile(optimizer = 'SGD',loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(X_train, Y_train,batch_size=32, epochs= 50,validation_split= 0.2 )

# acc  = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# epochs = range(len(acc)) 
# plt.figure(figsize = (15,10))
# plt.plot(epochs, acc,label= ['Training '])
# plt.plot(epochs, val_acc,label = ['Validation '])
# plt.legend()
model.save('model.hdf5')


# y_pred = model.predict(X_test)
# # y_pred = y_pred>0.5
# print (y_pred)

# # acc_score = accuracy_score(y_pred, Y_test)
# # print (f' acc_score = {acc_score}')
# # print (y_pred)