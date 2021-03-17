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
    ctypes.WinDLL(cuda_path+dll)

import numpy as np
import matplotlib.pylab as pl
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.preprocessing import MinMaxScaler

x = np.linspace(0, 2 * np.pi, 10000).reshape(-1,1)
y = np.sin(x)

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X = x_scaler.fit_transform(x)
Y = y_scaler.fit_transform(y)

model = Sequential()
model.add(Dense(16, activation='relu', kernel_initializer='he_uniform', input_dim=1))
model.add(Dense(16, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='linear', kernel_initializer='he_uniform'))
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

model.fit(X, Y, epochs=20, batch_size=32)

Y_test = model.predict(X, batch_size=32)

Y_res = y_scaler.inverse_transform(Y)
Y_test_res = y_scaler.inverse_transform(Y_test)

print(Y_res[0:10])
print(Y_test_res[0:10])

pl.subplot(211)
pl.plot(Y_res, label='ann')
pl.plot(Y_test_res, label='train')
pl.xlabel('#')
pl.ylabel('sin(x)')
pl.subplot(212)
pl.plot(Y_res - Y_test_res, label='diff')
pl.legend()
pl.show()

