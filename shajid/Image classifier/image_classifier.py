# ============================================================
# FAST Image Classification using ResNet50 on CIFAR-10
# Runs within ~30–60 minutes on GPU
# ============================================================

import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ------------------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------------------
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print("Dataset Loaded")

# ------------------------------------------------------------
# 2. Optimized tf.data Pipeline
# ------------------------------------------------------------
IMAGE_SIZE = 128        
BATCH_SIZE = 64        

def preprocess(image, label):
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = preprocess_input(image)
    return image, label

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = (
    train_ds
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(1000)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_ds = (
    test_ds
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

print("Data Pipeline Ready")

# ------------------------------------------------------------
# 3. Load Pretrained ResNet50
# ------------------------------------------------------------
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
)

base_model.trainable = False  

# ------------------------------------------------------------
# 4. Custom Classification Head (LIGHT)
# ------------------------------------------------------------
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)  
output = Dense(10, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# ------------------------------------------------------------
# 5. Compile Model
# ------------------------------------------------------------
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ------------------------------------------------------------
# 6. Callbacks (AUTO STOP)
# ------------------------------------------------------------
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=2,
    restore_best_weights=True
)

# ------------------------------------------------------------
# 7. Train Model (FAST)
# ------------------------------------------------------------
history = model.fit(
    train_ds,
    epochs=5,                   
    validation_data=test_ds,
    callbacks=[early_stop]
)

# ------------------------------------------------------------
# 8. Evaluate Model
# ------------------------------------------------------------
test_loss, test_acc = model.evaluate(test_ds)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

# ------------------------------------------------------------
# 9. Plot Results
# ------------------------------------------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title("Accuracy")

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Loss")

plt.show()
