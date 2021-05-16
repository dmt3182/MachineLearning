import tensorflow
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D,MaxPool2D,Flatten,Dense,Dropout
from tensorflow.keras.models import Sequential

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

print (X_test.shape,Y_test.shape)
classes = ['airplane','automobile','bird','cat','dear','dog','frog','horse','ship','truck']
classes[Y_train[1][0]]

#visualizing the data
plt.imshow(X_train[8])
classes[Y_train[8][0]]

# normalizing the data (0,1)
# pixel 0-255
X_train, X_test = X_train/255.0,X_test/255.0

# converting to categorical data
Y_train, Y_test = to_categorical(Y_train), to_categorical(Y_test)

#1. model building
#2. compile
#3. fit
model = Sequential()
model.add(Conv2D(32,3,activation='relu',padding  = 'same' , kernel_initializer= 'he_uniform'))
model.add(MaxPool2D(2,2))

model.add(Conv2D(64,3,activation='relu',padding  = 'same',kernel_initializer= 'he_uniform'))
model.add(MaxPool2D(2,2))

model.add(Conv2D(128,3,activation='relu',padding  = 'same',kernel_initializer= 'he_uniform'))
model.add(MaxPool2D(2,2))

model.add(Flatten())

model.add(Dense(128,activation = 'relu'))
model.add(Dropout(0.4))

model.add(Dense(64,activation = 'sigmoid'))
model.add(Dropout(0.2))

model.add(Dense(32,activation = 'relu'))
model.add(Dropout(0.6))

model.add(Dense(10,activation = 'softmax'))
model.add(Dropout(0.2))



model.compile('SGD', loss= 'categorical_crossentropy', metrics= ['accuracy'])
model.fit(X_train, Y_train, epochs  = 15, batch_size = 32)
y_pred = model.predict(X_test)
print(y_pred)