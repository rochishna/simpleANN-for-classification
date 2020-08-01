import numpy as np
import os ; os.environ['HDF5_DISABLE_VERSION_CHECK']='2'
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('bankdb.csv')
X = dataset.iloc[, 313].values
y = dataset.iloc[, 13].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[, 1] = labelencoder_X_1.fit_transform(X[, 1])
labelencoder_X_2 = LabelEncoder()
X[, 2] = labelencoder_X_2.fit_transform(X[, 2])
onehotencoder = OneHotEncoder()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
import keras
from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 10))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 10, epochs = 50)
y_pred = classifier.predict(X_test)
y_pred = (y_pred  0.5)
new_prediction = classifier.predict(sc.transform(np.array([[300, 1, 20, 3, 600, 2, 1, 1, 500,1]])))
n=new_prediction
new_prediction = (new_prediction  0.5)
print(n[0][0])
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
