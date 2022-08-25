from email.mime import base
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow import keras
from keras import layers
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocessing
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import set_random_seed


SEED = 42
DATASET_PATH = "../../../dataset/"
EPOCHS = 150
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
INPUT_SHAPE = (224, 224, 3, )


def set_seeds(seed):
    # `PYTHONHASHSEED` environment variable
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Python built-in random, numpy(+ scikit) and tensorflow seed
    set_random_seed(seed)


def plot_history(history, x_plot, name="plot.png"):
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
    os.makedirs("./Results", exist_ok=True)
    plt.savefig("./Results/"+name, dpi=100)
    plt.close()


def plot_confusionMatrix(labels, predictions, name="cm.png"):
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    print('True Negatives: ', cm[0][0])
    print('False Positives: ', cm[0][1])
    print('False Negatives: ', cm[1][0])
    print('True Positives: ', cm[1][1])
    os.makedirs("./Results", exist_ok=True)
    plt.savefig("./Results/"+name, dpi=100)
    plt.close()
        

def create_model(tipo):
    model = None
    if tipo == "scratch_model":
        model = keras.Sequential(
            [
                keras.Input(shape=(INPUT_SHAPE)),
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
    else:
        base_mobilenet = MobileNetV2(weights="imagenet", include_top=False, input_shape=INPUT_SHAPE)
        base_mobilenet.trainable = False    # Freeze weights
        inputs = keras.Input(shape=INPUT_SHAPE)
        x = mobilenet_preprocessing(inputs)
        x = base_mobilenet(x, training=False)
        x = layers.BatchNormalization()(x),
        x = layers.MaxPooling2D(pool_size=(2, 2))(x),
        x = layers.Flatten()(x),
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2, seed=SEED)(x)  # Regularize with dropout
        outputs = layers.Dense(2, activation="softmax")(x)
        model =  keras.Model(inputs, outputs)
        
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
            layers.RandomContrast((0.2, 0.2), seed=SEED),
            layers.Lambda(lambda x: tf.image.stateless_random_brightness(x, 0.2, seed=(SEED, SEED)))
        ]
    )
    # Rescale all datasets to [0.-1.]
    dataset = dataset.map(lambda x, y: (rescale(x), y), num_parallel_calls=AUTOTUNE)

    # Data augmentation only on the training set
    if augment: dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)

    # Use buffered prefetching on all datasets
    return dataset.cache().prefetch(buffer_size=AUTOTUNE)


def train(model, train_dataset, validation_dataset, name):
    print(f"Starting training {name}...")
    # Create some callbacks to avoid overfitting
    early_stopping = EarlyStopping(monitor="val_loss", patience=17, min_delta=0.0001, restore_best_weights=True, verbose=1)
    lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.001, patience=5, verbose=1)
    callbacks = [early_stopping, lr_scheduler]
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
        )
    # Train the model
    history = model.fit(train_dataset, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callbacks, validation_data=validation_dataset, workers=-1)
    # Save the model
    os.makedirs("./"+name, exist_ok=True)
    model.save("./"+name+"/"+name)
    # Evaluate the model
    score = model.evaluate(validation_dataset)
    print(f"Val Loss: {score[0]}")
    print(f"Val Accuracy: {score[1]}")
    
    x_plot = list(range(1, len(history.history['val_accuracy']) + 1))
    plot_history(history, x_plot, name=name+"_plot.png")


def test(model, test_dataset, name):
    X_test, y_test = [], []
    for image, label in test_dataset:
        image /= 255.
        X_test.append(image.numpy())
        y_test.append(label.numpy())
         
    X_test = tf.convert_to_tensor(np.asarray(X_test, dtype='float32'))
    y_test = np.asarray(y_test, dtype='float32')
    
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.batch(BATCH_SIZE)
    
    # Evaluate the model
    score = model.evaluate(test_dataset)
    print(f"Test Loss: {score[0]}")
    print(f"Test Accuracy: {score[1]}")
    
    predictions = model.predict(X_test)   
    predictions = np.argmax(predictions, axis=1)
    y_test = np.argmax(y_test, axis=1)
    print(classification_report(y_test, predictions))
    plot_confusionMatrix(y_test, predictions, name=name+"_cm.png")


def main():
    run_name = ["scratch_model", "fine_tuned"]
    
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
    # Load the test dataset
    print("Loading test dataset...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        directory=os.path.join(DATASET_PATH, "test"),
        label_mode="categorical",
        validation_split=None,
        image_size=(224, 224),
        batch_size=None,
        seed=SEED
        )
    print("Test dataset loaded!")
    print("Preparing datasets...")
    train_ds = prepare_dataset(train_ds, augment=True)
    val_ds = prepare_dataset(val_ds)
    print("Datasets prepared!")
    
    for name in run_name:
        print(f"Creating model {name}...")
        model = create_model(name)
        print("Model created!")
        
        train(model, train_ds, val_ds, name)
        test(model, test_ds, name)


if __name__ == "__main__":
    set_seeds(SEED)
    main()