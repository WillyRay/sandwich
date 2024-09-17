import sys
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("./data/sim_data.csv", index_col=["run"]).sort_index()

counts = []
cdffs = []
occupancies = []
anyCps = []
decays = []
touchTransferFractions = []



for run in data.index.unique():
    decays.append(data.loc[run]["decayRate"][0:1].values[0])
    touchTransferFractions.append(data.loc[run]["surfaceTransferFraction"][0:1].values[0])
    counts.append([data.loc[run]["count"]])
    cdffs.append([data.loc[run]["CDIFF"]])
    occupancies.append([data.loc[run]["occupancy"]])
    anyCps.append([data.loc[run]["anyCP"]])

X = np.stack((np.array(counts), np.array(cdffs), np.array(occupancies), np.array(anyCps)), axis=1)
y = np.stack((np.array(decays), np.array(touchTransferFractions)), axis=1)

# Load your data
# Assuming each time series is a list of numbers
counts = np.array(counts)
cdffs = np.array(cdffs)
occupancies = np.array(occupancies)
anyCps = np.array(anyCps)
decays = np.array(decays)
touchTransferFractions = np.array(touchTransferFractions)

# Stack your 4 time series horizontally
X = np.stack((counts, cdffs, occupancies, anyCps), axis=-1)

# Your 2 values to predict
y = np.stack((decays, touchTransferFractions), axis=-1)


# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""
# Reshape your data to 2D
X_train_2D = X_train.reshape(X_train.shape[0], -1)
X_test_2D = X_test.reshape(X_test.shape[0], -1)

# Apply StandardScaler
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train_2D)
X_test_scaled = scaler_X.transform(X_test_2D)

print(X_train_scaled.shape)


scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Reshape input to be [samples, time steps, features]
X_train_scaled = np.reshape(X_train_scaled, (X_train_scaled.shape[0], X_train_scaled.shape[1], 4))
X_test_scaled = np.reshape(X_test_scaled, (X_test_scaled.shape[0], X_test_scaled.shape[1], 4))
# Reshape input to be [samples, time steps, features]

"""
# Define the model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(2))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=10, verbose=0)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print("Test Loss:", loss)

# Make predictions
predictions = model.predict(X_test)
# Inverse transform the scaled prediction back to their original scale
predictions = y.inverse_transform(predictions)
print("Predicted Parameters:", predictions)




