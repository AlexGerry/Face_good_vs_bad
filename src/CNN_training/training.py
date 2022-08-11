import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from keras import layers
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocessing
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report


SEED = 42
DATASET_PATH = "../../../dataset/"
EPOCHS = 50
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE


def plot_history(history, x_plot):
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(x_plot, history.history['loss'])
    plt.plot(x_plot, history.history['val_loss'])
    plt.legend(['Training', 'Validation'])

    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(x_plot, history.history['accuracy'])
    plt.plot(x_plot, history.history['val_accuracy'])
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.show()
    

def create_model():
    model = keras.Sequential(
        [
            keras.Input(shape=(224, 224, 3, )),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu", strides=(1, 1)),
            #layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu", strides=(1, 1)),
            #layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(128, kernel_size=(3, 3), activation="relu", strides=(1, 1)),
            #layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(256, kernel_size=(3, 3), activation="relu", strides=(1, 1)),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3, seed=SEED),
            layers.Dense(64, activation='relu'),
            layers.Dense(2, activation="softmax")
        ]
    )
    print(model.summary())
    return model


def prepare_dataset(dataset, augment=False):
    rescale = tf.keras.Sequential(
        [layers.Rescaling(1./255)]
    )
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical", seed=SEED),
            layers.RandomRotation(np.round(np.random.rand(), 1), fill_mode="constant", seed=SEED),
        ]
    )
    # Rescale all datasets to [0.-1.]
    dataset = dataset.map(lambda x, y: (rescale(x), y), num_parallel_calls=AUTOTUNE)

    # Data augmentation only on the training set
    if augment: dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)

    # Use buffered prefetching on all datasets
    return dataset.cache().prefetch(buffer_size=AUTOTUNE)


def training(train_dataset, validation_dataset):
    print("Creating base model...")
    model = create_model()
    print("Base model created!")
    # Create some callbacks to avoid overfitting
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, min_delta=0.0001, restore_best_weights=True, verbose=1)
    lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.001, patience=7, verbose=1)
    callbacks = [early_stopping, lr_scheduler]
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adamax(learning_rate=0.001),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
        )
    # Train the model
    history = model.fit(train_dataset, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callbacks, validation_data=validation_dataset, workers=-1)
    # Save the model
    os.makedirs("./Scratch_CNN", exist_ok=True)
    model.save("././Scratch_CNN/Baseline_CNN")
    # Evaluate the model
    score = model.evaluate(validation_dataset)
    print(f"Val Loss: {score[0]}")
    print(f"Val Accuracy: {score[1]}")
    #predictions = (model.predict(X_test) > 0.5).astype(np.int8)
    #print(classification_report(Y_test, predictions))
    return history


def test():
    pass


def main():
    # Load the training dataset
    print("Loading train dataset...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory=os.path.join(DATASET_PATH, "train"),
        label_mode="categorical",
        validation_split=None,
        image_size=(224, 224),
        batch_size=BATCH_SIZE,
        seed=SEED
        )
    print("Train dataset loaded!")
    print("Labels in the dataset: ", train_ds.class_names)
    #for image_batch, labels_batch in train_ds:
    #    print("Image batch shape: ", image_batch.shape)
    #    print(image_batch)  # can call .numpy() to convert to numpy.ndarray
    #    print("Labels batch shape: ", labels_batch.shape)
    #    print(labels_batch) # can call .numpy() to convert to numpy.ndarray
    #    break
    
    # Load the validation dataset
    print("Loading validation dataset...")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        directory=os.path.join(DATASET_PATH, "valid"),
        label_mode="categorical",
        validation_split=None,
        image_size=(224, 224),
        batch_size=BATCH_SIZE,
        seed=SEED
        )
    print("Validation dataset loaded!")
    print("Preparing datasets...")
    train_ds = prepare_dataset(train_ds, augment=True)
    val_ds = prepare_dataset(val_ds)
    print("Datasets prepared!")
    
    train_history = training(train_ds, val_ds)
    x_plot = list(range(1, len(train_history.history['val_accuracy']) + 1))
    plot_history(train_history, x_plot)
    
    test()


if __name__ == "__main__":
    main()