# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
df = pd.read_csv('Churn_Modelling.csv')
X = df.iloc[:, 3:-1].values
y = df.iloc[:, -1].values

# Encoding the categorical data

# 1) Encoding the Gender Column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# 2) Encoding the Geography column 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoded', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
# print(X)

# spliting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# print(X_train)

# Building the ANN
# import tensorflow as tf
import keras
from keras import layers

# 1) Initializing ANN
ann = keras.models.Sequential()

# 2) Adding input layer and first hidden layer
ann.add(layers.Dense(units=6, activation='relu'))

# 3) Adding second hidden layer
ann.add(layers.Dense(units=7, activation='relu'))

# 4) Adding the output layer
ann.add(layers.Dense(units=1, activation='sigmoid'))

# Training the ANN

# 1) Compiling the ANN
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 2) Training on dataset
ann.fit(X_train, y_train, batch_size=32, epochs=100)

# Making Predictions and evaluating the model

# 1) Predicting the result of a single observation

# Use our ANN model to predict if the customer with the following informations will leave the bank:
# Geography: France
# Credit Score: 600
# Gender: Male
# Age: 40 years old
# Tenure: 3 years
# Balance: $ 60000
# Number of Products: 2
# Does this customer have a credit card? Yes
# Is this customer an Active Member: Yes
# Estimated Salary: $ 50000
# So, should we say goodbye to that customer?

X_Que = np.array([[600, 'France', 'Male', 40, 3, 60000, 2, 1, 1, 50000]])
X_Que[:, 2] = le.transform(X_Que[:, 2])
X_Que = np.array(ct.transform(X_Que))
print(ann.predict(sc.transform(X_Que)) > 0.5)

# 2) Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)

# 3) Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))