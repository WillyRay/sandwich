import sys
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.model_selection import train_test_split

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

counts = np.array(counts)
cdffs = np.array(cdffs)
occupancies = np.array(occupancies)
anyCps = np.array(anyCps)
decays = np.array(decays)
touchTransferFractions = np.array(touchTransferFractions)

X = np.stack((counts, cdffs, occupancies, anyCps), axis=-1)
y = np.stack((decays, touchTransferFractions), axis=-1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.reshape(-1, 276, 4)
X_test = X_test.reshape(-1, 276, 4)

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(276,4)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(2))

model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=10, verbose=0)

loss = model.evaluate(X_test, y_test)
print("Test Loss:", loss)

predictions = model.predict(X_test)
print("Predicted Parameters:", predictions)

print(X_test)
print(y_test)
