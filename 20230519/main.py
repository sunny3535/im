from tensorflow.keras.datasets import mnist   

# the data, split between train and validation sets
(x_train, y_train), (x_valid, y_valid) = mnist.load_data()
print(x_train.shape)
print(x_valid.shape)
print(x_train.dtype)
print(x_train.min())
print(x_train.max())
print(x_train[0])


#matplotlib 설치
import matplotlib.pyplot as plt

image = x_train[0]
plt.imshow(image, cmap='gray')

print(y_train[0])

x_train = x_train.reshape(60000, 784)
x_valid = x_valid.reshape(10000, 784)

print(x_train.shape)
print(x_train[0])

x_train = x_train / 255
x_valid = x_valid / 255 

print(x_train.dtype)

print(x_train.min())

print(x_train.max())



import tensorflow.keras as keras
num_categories = 10

y_train = keras.utils.to_categorical(y_train, num_categories)
y_valid = keras.utils.to_categorical(y_valid, num_categories)

print(y_train[0:9])

from tensorflow.keras.models import Sequential

model = Sequential()

from tensorflow.keras.layers import Dense

model.add(Dense(units=512, activation='relu', input_shape=(784,)))
model.add(Dense(units = 512, activation='relu'))
model.add(Dense(units = 10, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    x_train, y_train, epochs=5, verbose=1, validation_data=(x_valid, y_valid)
)

import numpy as np
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt

m = -2  # -2 to start, change me please
b = 40  # 40 to start, change me please

# Sample data
x = np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9])
y = np.array([10, 20, 25, 30, 40, 45, 40, 50, 60, 55])
y_hat = x * m + b

plt.plot(x, y, '.')
plt.plot(x, y_hat, '-')
plt.show()

print("Loss:", np.sum((y - y_hat)**2)/len(x))