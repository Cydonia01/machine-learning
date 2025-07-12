import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from keras.utils import load_img, img_to_array
from keras.utils import image_dataset_from_directory
from keras import layers, Sequential

# Constants
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 25

# Load training set
train_ds = image_dataset_from_directory(
    'dataset/training_set',
    labels='inferred',
    label_mode='binary',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Load test set
test_ds = image_dataset_from_directory(
    'dataset/test_set',
    labels='inferred',
    label_mode='binary',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Data augmentation pipeline
data_augmentation = Sequential([
    layers.Rescaling(1./255),
    layers.RandomFlip("horizontal"),
    layers.RandomZoom(0.2),
    layers.RandomRotation(0.1)
])

# Build the CNN model
cnn = Sequential([
    data_augmentation,
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(*IMG_SIZE, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Flatten(),
    layers.Dense(units=128, activation='relu'),
    layers.Dense(units=1, activation='sigmoid')  # Binary classification
])

# Compile the model
cnn.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
cnn.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS
)

# Single Prediction
test_image = load_img(
    'dataset/single_prediction/cat_or_dog.jpg',
    target_size=IMG_SIZE
)
test_image_array = img_to_array(test_image) / 255.0  # Normalize
test_image_array = np.expand_dims(test_image_array, axis=0)  # Add batch dimension
result = cnn.predict(test_image_array)

# Retrieve class names from the training set
class_names = train_ds.class_names
print("Class indices:", {name: idx for idx, name in enumerate(class_names)})

# Predict class
prediction = class_names[1] if result[0][0] > 0.5 else class_names[0]
print(f'The predicted class is: {prediction}')
