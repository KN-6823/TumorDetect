import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set parameters
img_size = 150
batch_size = 32
epochs = 10

# Prepare data generators
train_datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'Data/brain_tumor_dataset',  # Path to training data
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary',  # Binary classification
    subset='training'  # Set as training data
)

validation_generator = train_datagen.flow_from_directory(
    'Data/brain_tumor_dataset',  # Path to training data
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary',  # Binary classification
    subset='validation'  # Set as validation data
)

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
print("Training the model...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=epochs
)

# Save the model
model.save('model/brain_tumor_model.h5')
print("Model saved successfully.")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model
with open('model/model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model saved successfully.")
