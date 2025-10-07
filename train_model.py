import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

# ------------------------------
# 1. Set random seed for reproducibility
# ------------------------------
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# ------------------------------
# 2. Define data parameters and directory
# ------------------------------
DATA_DIR = "./TrashNet"
TARGET_SIZE = (224, 224) # MobileNet expects 224x224 images
BATCH_SIZE = 32
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 5

# ------------------------------
# 3. Data Loading and Preprocessing using ImageDataGenerator
# ------------------------------
# Use a validation split so that 20% of data is used for validation.
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # 20% for validation
)

# For validation we only apply preprocessing
test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    seed=seed_value
)

validation_generator = test_datagen.flow_from_directory(
    DATA_DIR,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    seed=seed_value
)

# Store class names for later use in evaluation and UI.
class_names = list(train_generator.class_indices.keys())

# ------------------------------
# 4. Build the MobileNet-based Transfer Learning Model
# ------------------------------
# Load MobileNet without the top classification layers.
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the base model initially

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
num_classes = len(class_names)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

# Compile the model with an appropriate optimizer and loss function.
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ------------------------------
# 5. Train the Custom Classification Head
# ------------------------------
history = model.fit(
    train_generator,
    epochs=INITIAL_EPOCHS,
    validation_data=validation_generator
)

# ------------------------------
# 6. Evaluate the Model on the Validation Set
# ------------------------------
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Advanced Evaluation: Compute precision, recall, f-score, and confusion matrix.
# (We loop over the validation generator to fetch predictions.)
val_steps = validation_generator.samples // validation_generator.batch_size
y_true = []
y_pred = []
for i in range(val_steps):
    x_batch, y_batch = next(validation_generator)
    predictions = model.predict(x_batch)
    y_true.extend(np.argmax(y_batch, axis=1))
    y_pred.extend(np.argmax(predictions, axis=1))

# Classification Report
report = classification_report(y_true, y_pred, target_names=class_names)
print("Classification Report:")
print(report)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# ------------------------------
# 7. Plot Training History (Accuracy and Loss)
# ------------------------------
plt.figure(figsize=(12,6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs. Validation Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(12,6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs. Validation Loss')
plt.legend()
plt.show()

# ------------------------------
# 8. Fine-Tuning MobileNet (Optional)
# ------------------------------
# Unfreeze the last 20 layers of MobileNet to fine-tune the model.
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Re-compile the model with a low learning rate.
model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_fine = model.fit(
    train_generator,
    epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
    initial_epoch=INITIAL_EPOCHS,
    validation_data=validation_generator
)

# ------------------------------
# 9. Save the Final Model to Disk
# ------------------------------
model.save("waste_classifier_mobilenet.h5")
print("Model saved as waste_classifier_mobilenet.h5")