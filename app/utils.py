import os
from pathlib import Path

import numpy as np #for arrays
import pandas as pd #for dataframe

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


BASE_DIR = Path(__file__).resolve().parent.parent
CSV_DIR = os.path.join(BASE_DIR, 'app\insurance.csv')
insurance_dataset = pd.read_csv(CSV_DIR)


def health_cost_predictor(age, sex, bmi, children, smoker, region):
    #preprocessing of data
    #encoding categorical columns
    insurance_dataset.replace({'sex':{'male':0,'female':1}},inplace=True)
    insurance_dataset.replace({'smoker':{'yes':0,'no':1}},inplace=True)
    insurance_dataset.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}}, inplace=True)

    x=insurance_dataset.drop(columns='charges',axis=1)
    y=insurance_dataset['charges']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

    #loading linear regression model
    regressor=LinearRegression()    

    #fitting the line
    regressor.fit(x_train.values,y_train.values)

    #evaluating the model
    #r squared vale is a statisitcal tool that works using variance where an dependent variable depends on an independent variable
    #to prevent overfitting checking my model on both test and train dataset

    # prediction on training data
    training_data_prediction =regressor.predict(x_train.values)

    # R squared value
    r2_train = metrics.r2_score(y_train, training_data_prediction)

    # prediction on test data
    test_data_prediction =regressor.predict(x_test.values)

    # R squared value
    r2_test = metrics.r2_score(y_test, test_data_prediction)

    #building the predictive system
    #input_data=(60,1,25.84,0,1,3)

    input_data = (age, sex, bmi, children, smoker, region)

    # changing input_data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = regressor.predict(input_data_reshaped)
    
    return prediction[0]