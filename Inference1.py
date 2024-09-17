import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

# Creating new Object for StandardScaler
sc = StandardScaler()

model = keras.models.load_model('ANN.h5')
print("Model Loaded>>>>>>>>>>>>>>>>")
def exit_prediction(credit_score,age,tenure,balance, num_products,hascard,is_active_member,estimated_salary,germany,spain,male):
    print("Transformed Input : ",sc.fit_transform([[credit_score,age,tenure,balance, num_products,hascard,is_active_member,estimated_salary,germany,spain,male]]))
    pred=model.predict(sc.fit_transform([[credit_score,age,tenure,balance, num_products,hascard,is_active_member,estimated_salary,germany,spain,male]]))
    print("Exiting Probability : ", pred[0][0])
    output=[1 if y > 0.3 else 0 for y in pred]
    print("%%%%%%% Output : ",output)

    if output[0]==1:
        print("Final Prediction : Customer will exit the bank")
    else:
        print("Final Prediction : Customer won't exit the bank")
exit_prediction(805, 56, 6, 151802, 1, 1, 0, 46791, 1, 0, 1)