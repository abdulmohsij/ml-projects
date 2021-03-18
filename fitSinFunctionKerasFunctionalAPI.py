import numpy as np
import matplotlib.pylab as pl
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

from sklearn.preprocessing import MinMaxScaler

x = np.linspace(0, 2 * np.pi, 10000).reshape(-1, 1)
y = np.sin(x)

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X = x_scaler.fit_transform(x)
Y = y_scaler.fit_transform(y)

input = Input(shape=(1,))
out = Dense(16, activation='relu', kernel_initializer='he_uniform')(input)
out = Dense(16, activation='relu', kernel_initializer='he_uniform')(out)
out = Dense(1, activation='linear', kernel_initializer='he_uniform')(out)
model = Model(inputs=input, outputs=out)
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

model.fit(X, Y, epochs=20, batch_size=32)

Y_test = model.predict(X, batch_size=32)

Y_res = y_scaler.inverse_transform(Y)
Y_test_res = y_scaler.inverse_transform(Y_test)

pl.subplot(211)
pl.plot(Y_res, label='ann')
pl.plot(Y_test_res, label='train')
pl.xlabel('#')
pl.ylabel('sin(x)')
pl.subplot(212)
pl.plot(Y_res - Y_test_res, label='diff')
pl.legend()
pl.show()

