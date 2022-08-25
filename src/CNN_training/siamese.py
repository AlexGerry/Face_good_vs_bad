import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
import itertools
from pathlib import Path
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet
from random import sample
import skimage
from tqdm import tqdm

SEED = 42
DATASET_PATH = "C:/Users/Alessandro/Desktop/università/visual image/dataset/"
BATCH_SIZE = 8
target_shape = (200,200)

class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)

class SiameseModel(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=0.5):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]

def set_seeds(SEED):
    os.environ['PYTHONHASHSEED'] = str(SEED)
    tf.keras.utils.set_random_seed(SEED)


def load_data(path):
    positive_path= path + "savory/"
    negative_path= path + "unsavory/"
    positive_images = [str(positive_path +"/"+ f) for f in os.listdir(positive_path)]
    negative_images = [str(negative_path +"/"+ f) for f in os.listdir(negative_path)]
    anchor = positive_images
    return positive_images, negative_images, anchor

def preprocess_image(filename):
    """
    Load the specified file as a JPEG image, preprocess it and
    resize it to the target shape.
    """

    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_shape)
    return image


def preprocess_triplets(anchor, positive, negative):
    """
    Given the filenames corresponding to the three images, load and
    preprocess them.
    """

    return (
        preprocess_image(anchor),
        preprocess_image(positive),
        preprocess_image(negative),
    )

def main():
    train_path = DATASET_PATH + "train/"
    valid_path = DATASET_PATH + "valid/"
    
    positive, negative, anchor = load_data(train_path)
    np.random.shuffle(negative)
    np.random.shuffle(anchor)
    triples = (anchor,positive,negative)
    train = tf.data.Dataset.from_tensor_slices(triples)
    #train = tf.data.Dataset.zip(train)
    train = train.shuffle(buffer_size=512, seed=SEED)
    train = train.map(preprocess_triplets)
    train = train.batch(32, drop_remainder=False)
    train = train.prefetch(8)

    v_pos, v_neg, v_anch = load_data(valid_path)
    np.random.shuffle(v_neg)
    np.random.shuffle(v_anch)
    triples = (v_anch, v_pos, v_neg)
    valid = tf.data.Dataset.from_tensor_slices(triples)
    #valid = tf.data.Dataset.zip(valid)
    valid = valid.shuffle(buffer_size=512, seed=SEED)
    valid = valid.map(preprocess_triplets)
    valid = valid.batch(32, drop_remainder=False)
    valid = valid.prefetch(8)
    
    base_cnn = resnet.ResNet50(weights="imagenet", input_shape=target_shape + (3,), include_top=False)

    flatten = layers.Flatten()(base_cnn.output)
    dense1 = layers.Dense(512, activation="relu")(flatten)
    dense1 = layers.BatchNormalization()(dense1)
    dense2 = layers.Dense(256, activation="relu")(dense1)
    dense2 = layers.BatchNormalization()(dense2)
    output = layers.Dense(256)(dense2)

    embedding = Model(base_cnn.input, output, name="Embedding")

    trainable = False
    for layer in base_cnn.layers:
        if layer.name == "conv5_block1_out":
            trainable = True
        layer.trainable = trainable

    anchor_input = layers.Input(name="anchor", shape=target_shape + (3,))
    positive_input = layers.Input(name="positive", shape=target_shape + (3,))
    negative_input = layers.Input(name="negative", shape=target_shape + (3,))

    distances = DistanceLayer()(
        embedding(resnet.preprocess_input(anchor_input)),
        embedding(resnet.preprocess_input(positive_input)),
        embedding(resnet.preprocess_input(negative_input)),
    )

    siamese_network = Model(inputs=[anchor_input, positive_input, negative_input], outputs=distances)

    siamese_model = SiameseModel(siamese_network)
    siamese_model.compile(
        optimizer=optimizers.Adam(0.0001),
            weighted_metrics=[])
    siamese_model.fit(train, epochs=4, validation_data=valid, batch_size=BATCH_SIZE)

    sample = next(iter(train))
    anchor, positive, negative = sample
    anchor_embedding, positive_embedding, negative_embedding = (
    embedding(resnet.preprocess_input(anchor)),
    embedding(resnet.preprocess_input(positive)),
    embedding(resnet.preprocess_input(negative)),
    )
    cosine_similarity = metrics.CosineSimilarity()

    positive_similarity = cosine_similarity(anchor_embedding, positive_embedding)
    print("Positive similarity:", positive_similarity.numpy())

    negative_similarity = cosine_similarity(anchor_embedding, negative_embedding)
    print("Negative similarity", negative_similarity.numpy())


if __name__ == "__main__":
    set_seeds(SEED)
    main()
