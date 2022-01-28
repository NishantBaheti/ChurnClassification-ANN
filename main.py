# -*- coding: utf-8 -*-
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import pandas as pd

completeFilePath = os.getcwd() + '/Churn_Modelling.csv'

df = pd.read_csv(completeFilePath)

# data preprocessing
featureColumn = ['CreditScore', 'Geography',
                 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
                 'IsActiveMember', 'EstimatedSalary']

targetColumn = ['Exited']
x = df[featureColumn].values
y = df[targetColumn].values

le_1 = LabelEncoder()
le_2 = LabelEncoder()
x[:, 1] = le_1.fit_transform(x[:, 1])
x[:, 2] = le_1.fit_transform(x[:, 2])

ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)

ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

ann.fit(x=x_train, y=y_train, batch_size=32, epochs=1000, verbose=1)

y_pred = ann.predict(x=x_test)

print(accuracy_score(y_test, y_pred > 0.5))

print(confusion_matrix(y_test, y_pred > 0.5))