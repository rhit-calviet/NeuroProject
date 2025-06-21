import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
import itertools

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Paths and settings
base_dir = 'AlzheimerDetection/DataProcessing/DATA'
train_dir = os.path.join(base_dir, 'TRAIN')
val_dir = os.path.join(base_dir, 'VAL')
test_dir = os.path.join(base_dir, 'TEST')
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 4
AUTOTUNE = tf.data.AUTOTUNE

# Create tf.data datasets
def prepare_dataset(directory):
    return tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode='categorical',
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True
    ).prefetch(buffer_size=AUTOTUNE)

train_dataset = prepare_dataset(train_dir)
val_dataset = prepare_dataset(val_dir)
# Load test dataset (non-shuffled, needed for evaluation and metrics)
raw_test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Prefetch for performance
test_dataset = raw_test_dataset.prefetch(buffer_size=AUTOTUNE)

# Get class names before prefetching
class_names = raw_test_dataset.class_names

# Create VGG16-based model
def create_vgg16_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Instantiate the model
vgg16_model = create_vgg16_model()

# Unfreeze last few conv layers in base model
for layer in vgg16_model.layers[0].layers[-3:]:
    layer.trainable = True

# Compile the model
vgg16_model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Callbacks
os.makedirs('models', exist_ok=True)
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6),
    ModelCheckpoint(
        filepath='models/vgg16_alzheimers_model.h5',
        monitor='val_accuracy',
        save_best_only=True
    ),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs/vgg16_training',
        histogram_freq=1,
        profile_batch='500,520'
    )
]

# Train the model
with tf.device('/GPU:0'):
    history = vgg16_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )

# Save final model
vgg16_model.save('models/vgg16_alzheimers_final.h5')

# Evaluate on test dataset
test_loss, test_acc, test_prec, test_rec = vgg16_model.evaluate(test_dataset)
print(f"\nTest Accuracy:  {test_acc * 100:.2f}%")
print(f"Test Precision: {test_prec * 100:.2f}%")
print(f"Test Recall:    {test_rec * 100:.2f}%")

# Generate predictions
y_pred = np.argmax(vgg16_model.predict(test_dataset), axis=1)
y_true = np.concatenate([np.argmax(label.numpy(), axis=1) for _, label in test_dataset])

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix'):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f'{cm[i, j]:.2f}' if normalize else f'{cm[i, j]}',
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

plot_confusion_matrix(cm, class_names, normalize=True, title="Normalized Confusion Matrix")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Training history
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Accuracy')
    plt.plot(val_acc, label='Val Accuracy')
    plt.legend()
    plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.legend()
    plt.title("Loss")
    plt.tight_layout()
    plt.show()

plot_training_history(history)
