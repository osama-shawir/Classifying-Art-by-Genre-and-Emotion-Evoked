import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from tensorflow.keras import backend as K
import shutil
import random
import sys

# Make sure that the script is run on a GPU
if not tf.config.list_physical_devices("GPU"):
    print("No GPU was detected. This script requires a GPU to run.")
    sys.exit()

# Make the results reproducible to compare later
seed_value = 42

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ["PYTHONHASHSEED"] = str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

# 4. Set `tensorflow` pseudo-random generator at a fixed value
tf.random.set_seed(seed_value)

# Define paths to your datasets
base_dir = "./emotions"
train_val_dir = "./data_em/train_val"
test_dir = "./data_em/test"

# Split the data into train+validation and test sets
for emotion in os.listdir(base_dir):
    emotion_dir = os.path.join(base_dir, emotion)
    images = os.listdir(emotion_dir)
    train_val_images, test_images = train_test_split(
        images, test_size=0.2, random_state=42
    )  # 20% for testing

    # Create directories for each emotion in train_val_dir and test_dir
    train_val_emotion_dir = os.path.join(train_val_dir, emotion)
    test_emotion_dir = os.path.join(test_dir, emotion)
    os.makedirs(train_val_emotion_dir, exist_ok=True)
    os.makedirs(test_emotion_dir, exist_ok=True)

    for image in train_val_images:
        shutil.copy(
            os.path.join(emotion_dir, image), os.path.join(train_val_emotion_dir, image)
        )
    for image in test_images:
        shutil.copy(
            os.path.join(emotion_dir, image), os.path.join(test_emotion_dir, image)
        )

# Now you can use ImageDataGenerator to split train_val_dir into training and validation sets
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.1,
    zoom_range=0.1,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.3,  # 30% of train_val for validation
)

train_generator = train_datagen.flow_from_directory(
    train_val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    subset="training",  # set as training data
)

validation_generator = train_datagen.flow_from_directory(
    train_val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    subset="validation",  # set as validation data
)

# For the test set, you can use another ImageDataGenerator without data augmentation
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(224, 224), batch_size=32, class_mode="binary"
)


# Define a function to calculate F1 score
def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


# Add a new classification layer
model = tf.keras.models.load_model(
    "sim_genre_clf_94_0.57.keras", custom_objects={"f1_score": f1_score}
)
model.pop()
model.add(Dense(1, activation="sigmoid"))

# Extract the base model
base_model = model.layers[0]

# Freeze base model except conv5 layers of the base model
for layer in base_model.layers:
    if layer.name.startswith("conv5") or layer.name.startswith("input"):
        layer.trainable = True
    else:
        layer.trainable = False

# If there's a class imbalance, use class weights
class_weights = compute_sample_weight(
    class_weight="balanced", y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))

# Use Adam optimizer with the learning rate of 0.1
optimizer = Adam(learning_rate=0.000001)

# Include f1_score in the metrics list when compiling the model
model.compile(
    optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy", f1_score]
)

# Define ModelCheckpoint callback
model_checkpoint = ModelCheckpoint(
    "sim_genre_to_emotion_clf_{epoch}_{val_accuracy:.2f}_{val_f1_score:.2f}.keras",
    monitor="val_f1_score",
    mode="max",
    save_best_only=True,
)

history = model.fit(
    train_generator,
    epochs=100,
    validation_data=validation_generator,
    verbose=2,
    callbacks=model_checkpoint,
    class_weight=class_weights,
)

# Evaluate the model on the test set
print("Model evaluation on test set:")
model.evaluate(test_generator)

# Save the final model
model.save("sim_genre_to_emotion_clf_final.keras")
