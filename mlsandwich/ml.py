import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

real_data = pd.read_csv("./data/surfaces_by_day.csv", index_col=0)
real_counts = real_data['count']
#print(real_data)

simulated_data = pd.read_csv("./data/surfaces_indexed.csv", usecols=["run","tick", "count","decayRate","surfaceTransferFraction"]).sort_values(by=['run','tick'])
limited_to_last_55_days = simulated_data.query('tick > 299')



simulated_day_count_arrays = []
parameters = []
for i in simulated_data["run"].unique():
    parameters.append(simulated_data[simulated_data.run==i][["decayRate", "surfaceTransferFraction"]].iloc[0].to_numpy())
    simulated_day_count_arrays.append(list(simulated_data[simulated_data.run == i ][["tick","count"]].query('tick > 299')["count"]))


print(simulated_day_count_arrays)
print(parameters)


#print(len(np.ndarray(simulated_day_count_arrays)))
#print(len(parameters))

X = limited_to_last_55_days.drop(columns=["run", "tick", "decayRate", "surfaceTransferFraction"]).values
y = limited_to_last_55_days[["decayRate", "surfaceTransferFraction"]].values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




# Normalize input and output data
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

model = tf.keras.Sequential([
    tf.keras.layers.Dense(55, activation='relu', input_shape=(55,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(2)  # Assuming you have 2 parameters to predict
])
# Compile the model
model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Step 5: Evaluate your model
loss = model.evaluate(X_test, y_test)
print("Test Loss:", loss)

# Predict inputs based on new output values
#new_output = 10  # Example output value
#predicted_inputs_scaled = model.predict(scaler_y.transform([[new_output]]))
#predicted_inputs = scaler_X.inverse_transform(predicted_inputs_scaled)
#print("Predicted inputs for output", new_output, ":", predicted_inputs)
