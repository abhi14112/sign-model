import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
data = np.load('preprocessed_data.npz')
x_train, y_train = data['X'], data['Y']
x_train = np.array(x_train).reshape(-1, 98, 98, 1)
y_train = tf.keras.utils.to_categorical(y_train)
model = Sequential()
model.add(Conv2D(48, (3, 3), input_shape=(98, 98, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(48, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=156, activation='relu'))
model.add(Dropout(0.40))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.40))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=13, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print("Training...")
history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.3)
model.save('sign_model.h5')