import numpy as np
import matplotlib.pylab as pl
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.preprocessing import MinMaxScaler

# Generate the training set.
x = np.linspace(0, 2 * np.pi, 10000).reshape(-1,1)
y = np.sin(x)

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

# Transform the values.
# See: https://stackoverflow.com/questions/4674623/why-do-we-have-to-normalize-the-input-for-an-artificial-neural-network
X = x_scaler.fit_transform(x)
Y = y_scaler.fit_transform(y)

# Create a model using the Sequential API.
model = Sequential()
model.add(Dense(16, activation='tanh', kernel_initializer='he_uniform', input_dim=1))
model.add(Dense(16, activation='tanh', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='linear', kernel_initializer='he_uniform'))
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Train the model on our scaled data.
model.fit(X, Y, epochs=20, batch_size=32)

# Test the model on the training data to figure out the training accuracy.
# Traditionally, we would have a separate test set. In this case, we need to see how our estimated sin function is to the original.
Y_test = model.predict(X, batch_size=32)

# Convert the predictions back to the original scale.
Y_res = y_scaler.inverse_transform(Y)
Y_test_res = y_scaler.inverse_transform(Y_test)

# Plot the results and the difference.
pl.subplot(211)
pl.plot(Y_res, label='ann')
pl.plot(Y_test_res, label='train')
pl.legend()
pl.xlabel('x')
pl.ylabel('sin(x)')
pl.subplot(212)
pl.plot(Y_res - Y_test_res, label='diff')
pl.legend()
pl.show()

