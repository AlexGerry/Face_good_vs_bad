import keras_tuner as kt
from tensorflow import keras
from keras import layers, regularizers
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import os
import matplotlib.pyplot as plt
import seaborn as sns


INPUT_SHAPE = (224, 224, 3, )


def build_model(hp):
    model = keras.models.Sequential()
    model.add(keras.Input(shape=(INPUT_SHAPE)))
    
    # Search first conv
    model.add(layers.Conv2D(
        filters=hp.Choice('conv_1_filter', values=[32, 64]),
        kernel_size=hp.Choice('conv_1_kernel', values = [3,5]),
        activation='relu',
        kernel_regularizer=regularizers.l2(1e-2),
        strides=(1, 1)
    ))
    model.add(layers.MaxPooling2D(pool_size=hp.Choice('pool_1_size', values = [3,5])))
    
    # Choose how many conv layers
    for i in range(hp.Int("num_Convolutional_layers", 1, 2)):
        model.add(
            layers.Conv2D(
                filters=hp.Choice(f"conv_{i}_filters", values=[64, 128, 256]),
                kernel_size=(3, 3),
                activation="relu",
                kernel_regularizer=regularizers.l2(1e-2),
                strides=(1, 1)
            )
        )
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.BatchNormalization())

    # Choose how many classifier
    for i in range(hp.Int("num_FullyConnected_layers", 1, 2)):
        model.add(
            layers.Dense(
                # Tune number of units separately.
                units=hp.Choice(f"units_{i}", values=[64, 128, 256]),
                activation="relu",
                kernel_regularizer=regularizers.l2(1e-2)
            )
        )
        if hp.Boolean("dropout"): model.add(layers.Dropout(rate=0.25))
    
    model.add(layers.Dense(2, activation="softmax"))

    # Choose the optimizer
    hp_optimizer = hp.Choice('optimizer', values=['adam', 'rmsprop', 'adamax'], default = 'adamax')
    optimizer = tf.keras.optimizers.get(hp_optimizer)
    # Choose the learning rate
    optimizer.learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-2], default = 1e-3)
                                        
    model.compile(optimizer=optimizer, 
                    loss="categorical_crossentropy", 
                    metrics = ["accuracy"])

    return model


def prepare_dataset(dataset, seed, autotune, batch_size, augment=False):
    rescale = tf.keras.Sequential(
        [layers.Rescaling(1./255)]
    )
    
    # Rescale all datasets to [0.-1.]
    dataset = dataset.map(lambda x, y: (rescale(x), y), num_parallel_calls=autotune)

    # Data augmentation only on the training set
    if augment: 
        data_augmentation = tf.keras.Sequential(
            [
                layers.RandomFlip("horizontal_and_vertical", seed=seed),
                layers.RandomRotation(np.round(np.random.rand(), 1), fill_mode="constant", seed=seed),
                layers.RandomContrast((0.2, 0.2), seed=seed),
                layers.Lambda(lambda x: tf.image.stateless_random_brightness(x, 0.2, seed=(seed, seed)))
            ]
        )
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=autotune)

    # Use buffered prefetching on all datasets
    return dataset.batch(batch_size).cache().prefetch(buffer_size=autotune)


def plot_history(history, x_plot, name="plot.png"):
    os.makedirs("./Results", exist_ok=True)
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(x_plot, history.history['loss'])
    plt.plot(x_plot, history.history['val_loss'])
    plt.legend(['Training', 'Validation'])
    plt.savefig("./Results/"+name, dpi=100)
    plt.close()

    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(x_plot, history.history['accuracy'])
    plt.plot(x_plot, history.history['val_accuracy'])
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.savefig("./Results/"+name, dpi=100)
    plt.close()
    
    
def plot_confusionMatrix(labels, predictions, name="cm.png"):
    print(classification_report(labels, predictions))
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