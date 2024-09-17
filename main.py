# Importing Libraries

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
import joblib

# Data import & Preprocessing

data = pd.read_csv('Churn_Modelling.csv')
X = data.iloc[:, 3:-1]
y = data.iloc[:, -1]

geography = pd.get_dummies(X['Geography'], dtype=int, drop_first=True)
gender = pd.get_dummies(X['Gender'], dtype=int, drop_first=True)

X = pd.concat([X, geography, gender], axis=1)
X = X.drop(columns=['Geography', 'Gender'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ANN------------------------------------------------

classifier = Sequential()
classifier.add(Dense(units=6, kernel_initializer='he_uniform', activation='relu', input_dim=11))
classifier.add(Dense(units=6, kernel_initializer='he_uniform', activation='relu'))
classifier.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))
classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

model_history = classifier.fit(X_train, y_train, validation_split=0.33, batch_size=10, epochs=50)

print(classifier.summary())

y_pred = classifier.predict(X_test)
output=classifier.predict(sc.transform([[120,35,2,50000,1,1,1,100000,0,0,1]]))
# print([1 if y>0.3 else 0 for y in output])

from sklearn.metrics import confusion_matrix
y_pred2 = [1 if y>0.3 else 0 for y in y_pred]
cm = confusion_matrix(y_test, y_pred2)

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred2)
print(score*100)

"""
# summarize history for accuracy (PLOT)
print(model_history.history.keys())
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss (PLOT)
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
"""

# Exporting the trained model 
classifier.save("ANN.h5")

# Exporting the trained Standard Scaler Transformation
joblib.dump(sc, 'Scaler.save')