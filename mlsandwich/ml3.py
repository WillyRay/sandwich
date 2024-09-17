import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

from scikeras.wrappers import KerasRegressor
real_data = pd.read_csv("./data/surfaces_by_day.csv", index_col=0).join(
pd.read_csv("./data/observed_matrix_1.csv", index_col=0)).drop([0])

print(real_data)
sim_data = pd.read_csv("./data/sim_data.csv", index_col=["run", "tick"]).query('tick > 310').sort_index()

runs = list(range(0, 200))
ticks = list(range(310, 366))
new_index = pd.MultiIndex.from_product([runs, ticks], names=['run', 'tick'])
sim_data.reindex(new_index, fill_value=0)
print(sim_data)

y = []
X = []
current_run = -1
for i, tick in sim_data.index:
    if i > current_run:
        current_run = i
        y.append(sim_data.loc[current_run, 312].drop(["count"]).to_list())
        X.append(sim_data.loc[current_run, :]["count"].values.tolist())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1024)

# Convert lists to numpy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
print(X_train_scaled.shape)
print(X_test_scaled.shape)

data = real_data[1:56].values

# Reshape your input data to fit the requirements of Conv1D layers
X_train_reshaped = X_train_scaled.reshape(-1, 55,5, 1)
X_test_reshaped = X_test_scaled.reshape(-1, 55,5, 1)
sys.exit(0)

"""
# Define the neural network model with a Conv1D layer
# sklearn, gridsearch. Optimize toward certain metrics... MSE?  Raw accuracy?
def create_model(dense_neurons=32, learning_rate=0.01, filters=32, kernel_size=3):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=(55, 800)),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(dense_neurons, activation='relu'),
        tf.keras.layers.Dense(dense_neurons, activation='relu'),
        tf.keras.layers.Dense(2)  # Output layer with 2 neurons for parameters
    ])

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')

    return model


param_grid = {
    'dense_neurons': [32, 64, 128],
    'learning_rate': [0.01, 0.001, 0.0001],
    'filters': [32, 64, 128],
    'kernel_size': [3, 5, 7]
}
model = KerasRegressor(dense_neurons=32, learning_rate=0.01, filters=32, kernel_size=3, build_fn=create_model, verbose=0)
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(X_train_reshaped, y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

"""

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train_reshaped, y_train, epochs=100, batch_size=32, validation_data=(X_test_reshaped, y_test))
#plot loss per epoch, add earlyStopping, k-fold crossvalidation sklearn.model_selection.cross_validate
#filter size?

# Evaluate the model
loss = model.evaluate(X_test_reshaped, y_test)
print("Test Loss:", loss)

# Make predictions
data_reshaped = data.reshape(-1, 55, 1)
scaled_data_reshaped = scaler.transform(data_reshaped.reshape(-1, 1)).reshape(data_reshaped.shape)
predicted_parameters = model.predict(scaled_data_reshaped)
predicted_parameters = predicted_parameters.reshape(-1, 2)
print("Predicted Parameters:", predicted_parameters)
