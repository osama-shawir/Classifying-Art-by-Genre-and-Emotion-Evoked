import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from tensorflow.keras import backend as K
import shutil
from sklearn.model_selection import train_test_split
import sys, random

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
base_dir = "./genre"
train_val_dir = "./data/train_val"
test_dir = "./data/test"

# Split the data into train+validation and test sets
for genre in os.listdir(base_dir):
    genre_dir = os.path.join(base_dir, genre)
    images = os.listdir(genre_dir)
    train_val_images, test_images = train_test_split(
        images, test_size=0.2, random_state=42
    )  # 20% for testing

    # Create directories for each genre in train_val_dir and test_dir
    train_val_genre_dir = os.path.join(train_val_dir, genre)
    test_genre_dir = os.path.join(test_dir, genre)
    os.makedirs(train_val_genre_dir, exist_ok=True)
    os.makedirs(test_genre_dir, exist_ok=True)

    for image in train_val_images:
        shutil.copy(
            os.path.join(genre_dir, image), os.path.join(train_val_genre_dir, image)
        )
    for image in test_images:
        shutil.copy(os.path.join(genre_dir, image), os.path.join(test_genre_dir, image))

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
    class_mode="categorical",
    subset="training",  # set as training data
)

validation_generator = train_datagen.flow_from_directory(
    train_val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="validation",  # set as validation data
)

# For the test set, you can use another ImageDataGenerator without data augmentation
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(224, 224), batch_size=32, class_mode="categorical"
)

# Load pre-trained ResNet50 model without the top layer
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))


for layer in base_model.layers:
    if (
        layer.name.startswith("conv4")
        or layer.name.startswith("conv5")
        or layer.name.startswith("input")
    ):
        layer.trainable = True
    else:
        layer.trainable = False

# Add a new classification layer
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(4, activation="softmax"))  # 4 classes


def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


# ModelCheckpoint callback
model_checkpoint = ModelCheckpoint(
    "genre_l4_l5_{epoch}_{val_accuracy:.2f}_{val_f1_score:.2f}.keras",
    monitor="val_f1_score",
    mode="max",
    save_best_only=True,
)

# If there's a class imbalance, use class weights
class_weights = compute_sample_weight(
    class_weight="balanced", y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))

# Use Adam optimizer
optimizer = Adam(learning_rate=0.000001)

# Compile the model
model.compile(
    optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy", f1_score]
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
model.save("genre_l4_l5_final.keras")
