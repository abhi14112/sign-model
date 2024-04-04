import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
data = np.load('preprocessed_data.npz')
x_train, y_train = data['X'], data['Y']
x_train = np.array(x_train).reshape(-1, 90, 90, 1)
y_train = tf.keras.utils.to_categorical(y_train)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(90, 90, 1)),
    MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='same'),
    Dropout(0.2),
    Conv2D(64, (3, 3), activation='relu', strides=(2,2), padding='same'),  
    MaxPooling2D(pool_size = (2, 2), strides=(2,2), padding='same'),
    Dropout(0.4),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(13, activation='softmax')
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print("Training...")
history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.3)
model.save('my_cnn_model.h5')