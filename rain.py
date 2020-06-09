import tensorflow as tf
import pandas as pd

data = pd.read_csv(r'/home/viraj/Desktop/dataexport_20200423T201616.csv')
data = data.drop(data.index[[0,1,2,3,4,5,6,7,8,9]])# removing unnecessary rows
data.index = data.index-9

data.columns = ['variable','Tempertaure1','Temperature2','Temperature','Relative Humidity1','Relative Humidity2','Relative Humidity','Mean Sea Level Pressure1','Mean Sea Level Pressure2','Mean Sea Level Pressure','Precipitation Total','Cloud Cover Total','Wind Speed1','Wind Speed2','Wind Speed','Wind Direction']
data = data.drop(['variable'],axis = 1)
data = data.astype('float')

data.head()

y = data['Precipitation Total'] 
X = data.drop(['Precipitation Total'],axis = 1)

X.head()

import numpy as np

i = 0
y_label = np.zeros((4495,1),dtype = 'uint32')
for x in np.array(y,dtype = 'f'):
    if x>=0 and x<4:
        y_label[i] = 0;
    elif x>=4 and x<16:
        y_label[i] = 1;
    elif x>=16 and x<32:
        y_label[i] = 2;
    else:
        y_label[i] = 3;
    i = i+1
y_label

train_size = int(0.7*y.size)
test_size = int(0.3*y.size)

X1 = (X-X.mean())/X.std()

X_train = X1[0:train_size]
X_test = X1[train_size:int(y.size)]

X_train = np.ravel(X_train)
X_train = X_train.reshape(3146,14)
X_test = np.ravel(X_test)
X_test = X_test.reshape(1349,14)
y_label = np.ravel(y_label)
y_train = y_label[0:train_size]
y_test = y_label[train_size:int(y.size)]

def build_model():
    nn_model = tf.keras.Sequential([
    tf.keras.layers.Dense(14),
    tf.keras.layers.Dense(30,activation = tf.nn.relu),
    tf.keras.layers.Dense(30,activation = tf.nn.relu),
    tf.keras.layers.Dense(4,activation = tf.nn.softmax)
    ])
    return nn_model
model1 = build_model()

model1.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-2),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',np.unique(y_train),y_train)
# poor learning
model1.fit(X_train,y_train,batch_size = 10,epochs = 20,class_weight=class_weights,validation_split = 0.142857,verbose = 2)

results = model1.evaluate(X_test, y_test)

from imblearn.over_sampling import SMOTE

smote = SMOTE('minority')

Xt = np.ravel(X)
Xt = Xt.reshape(4495,14)
yt = np.ravel(y)
yt = yt.reshape(4495,1)

#balance class

X2,y2 = smote.fit_sample(Xt,y_label)
X3,y3 = smote.fit_sample(X2,y2)
X4,y4 = smote.fit_sample(X3,y3) 
from sklearn.model_selection import train_test_split as tts
X_train1,X_test1,y_train1,y_test1 = tts(X4,y4,train_size = 0.7,random_state=1)
model2 = build_model()

model2.compile(optimizer='adam',              
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model2.fit(X_train1,y_train1,batch_size = 10,epochs = 50,validation_split = 0.142857,verbose = 2)

pred = model2.predict(X_test1)
results = model2.evaluate(X_test1, y_test1)

@finding exact measure of rainfall

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
xtrain = sc.fit_transform(X_train)
xtest = sc.fit_transform(X_test)
xtrain = np.reshape(xtrain,(xtrain.shape[0],xtrain.shape[1],1))
xtest = np.reshape(xtest,(xtest.shape[0],xtest.shape[1],1))

y = np.ravel(y)
ytrain = y[0:train_size]
ytest = y[train_size:int(y.size)]

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.LSTM(units = 50, return_sequences = True, input_shape = (xtrain.shape[1],1)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(units = 50, return_sequences = True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(units = 50, return_sequences = True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(units = 50))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units=1))
model.compile(optimizer='adam',loss = 'mean_squared_error')
model.fit(xtrain,ytrain,epochs = 50,batch_size=32)

predx = xtest
predy = ytest

pred_rain = model.predict(predx)

pred_rain=np.reshape(pred_rain,(pred_rain.size,))

import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error as mse

mae = 0
i = 0
while(i<ytest.size):            # predicting rainfall of next 14 days
    plt.plot(model.predict(xtest[i:i+14]),color = 'b')
    plt.plot(predy[i:i+14], color = 'r')
    plt.show()
    if (i+14<ytest.size):
        inc=14
    else:
        ytest.size-i
    new = mse(predy[i:i+inc],model.predict(xtest[i:i+inc]))
    mae =  (mae*i + new*inc)/(i+inc)
    model.fit(xtest[i:i+14],ytest[i:i+14])
    i = i+14

print(mae)

