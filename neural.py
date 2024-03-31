import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout

data = np.load('preprocessed_data.npz')
x_train, y_train = data['X'], data['Y']

x_train = np.array(x_train).reshape(-1, 100, 100, 1)

# Convert labels to one-hot encoding

y_train = tf.keras.utils.to_categorical(y_train)
model = Sequential([
    Conv2D(16, (2, 2), activation='relu', input_shape=(100, 100, 1)),
    MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='same'),
    Conv2D(32, (3, 3), activation='relu', strides=(2,2), padding='same'),  # Adding another convolutional layer
    MaxPooling2D(pool_size = (3, 3), strides=(3,3), padding='same'),
    Conv2D(64, (5, 5), activation='relu'),  # Adding another convolutional layer
    MaxPooling2D(pool_size=(5, 5), strides=(5,5), padding='same'),
    Flatten(),
    Dense(120, activation='relu'),  # Adding a dense layer with more neurons
    Dropout(0.2),  # Adding dropout regularization
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Training...")
history = model.fit(x_train, y_train, epochs=15, batch_size=32, validation_split=0.2)

model.save('my_cnn_model.h5')