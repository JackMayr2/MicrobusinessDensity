import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Load our data
feature_df = pd.read_csv('Generated-Data/lagged_data.csv')

# Split data into feature / label columns
X, y = feature_df[
           ['Year', 'Month', 'pct_broadband', 'pct_college', 'pct_foreign_born', 'pct_it_workers', 'median_hh_income']], \
       feature_df[['microbusiness_density']]

X = (X - X.min()) / (X.max() - X.min())

print(X.info())
# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# For this py file, we will be creating a Sequential model. This is the simpler and faster way of first creating a model

# First, we will define the model
model = Sequential()

model.add(Dense(64,  activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

# Second, we will compile the model
model.compile(optimizer='sgd', loss='mse')

# Third, we will fit the model
model.fit(X_train, y_train, epochs=10, batch_size=128, verbose=1)

# Fourth, we will evaluate the model
error = model.evaluate(X_test, y_test, verbose=0)
print('MSE: %.3f, RMSE: %.3f' % (error, np.sqrt(error)))

# Finally, lets make a prediction. I will just be taking the first row of X
row = X.iloc[[0]]
yhat = model.predict([row])
print('Predicted: %.3f' % yhat)
