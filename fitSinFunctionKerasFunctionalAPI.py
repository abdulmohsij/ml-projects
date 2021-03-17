# :NOTE: 18-Mar-2021
# This CUDA loading wouldn't be necessary for a clean install of cuda and is only windows specific.
# For linux this section will need to be removed.
import ctypes

cuda_path = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.0\\bin\\"
to_load = ["cudart64_110.dll",
           "cublas64_11.dll",
           "cufft64_10.dll",
           "curand64_10.dll",
           "cusolver64_10.dll",
           "cusparse64_11.dll",
           "cudnn64_8.dll"]

for dll in to_load:
    ctypes.WinDLL(cuda_path + dll)

# end-windows-specific-section.

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

