import numpy as np
import tensorflow as tf
# Load the existing model
model = tf.keras.models.load_model('sign_model.h5')
# Define a new optimizer
new_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Example of creating a new Adam optimizer with a custom learning rate
# Compile the model with the new optimizer
model.compile(optimizer=new_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
# Load the new data
new_data = np.load('preprocessed_data.npz')
x_new, y_new = new_data['X'], new_data['Y']
x_new = np.array(x_new).reshape(-1, 98, 98, 1)
y_new = tf.keras.utils.to_categorical(y_new)
# Train the model on new data
print("Training on new data...")
history = model.fit(x_new, y_new, epochs=4, batch_size=32, validation_split=0.3)
# Optionally, evaluate the model
loss, accuracy = model.evaluate(x_new, y_new)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
# Save the fine-tuned model
model.save('fine_tuned_model.h5')