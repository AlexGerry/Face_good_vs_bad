import os
import pickle
from typing import Tuple
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras import models
from PIL import Image
from mtcnn.mtcnn import MTCNN
from matplotlib.pyplot import imread


class DeepModel(object):
    detector = MTCNN()
    
    def __init__(self, classifier_path:str, siamese_embeddings_path:str, image_size:Tuple=(200,200)) -> None:
        self.classifier = None
        self.siamese_embeddings = models.load_model(siamese_embeddings_path)
        self.target_shape = image_size
        with open(classifier_path, "rb") as f:
            self.classifier = pickle.load(f)
    
    
    def predict(self, image_path:str):
        """ This function predicts the label of the given image.
        
            Parameters:
            ----------
            image_path: str
                The path of the input image
                
            Returns:
            -------
            label: None
                The label of the predicted image
        
            """
        img = Image.open(image_path)
        img = img.resize(self.target_shape)
        feature = self.extract_features(img)
        return self.classifier.predict(feature)
    
    
    def extract_features(self, image):
        """ This function extract a feature vector from the given image.
        
            Parameters:
            ----------
            image: None
                The input image
                
            Returns:
            -------
            features: None
                The feature vector of the image
        
            """
        image = np.dstack([image])
        return [self.siamese_embeddings(tf.expand_dims(image, axis=0)).numpy()[0]]
    
    
    def cbir(self, image_path:str=None, k:int=10):
        """ This function outputs the most k similar image to the one given in input.
        
            Parameters:
            ----------
            image_path: str
                The path of the input image
            k: int
                The number of images to retrieve
                
            Returns:
            -------
            bho: None
                The k most similar images
        
            """
        pass
    
    
    @staticmethod
    def find_faces(image_path):
        res = []
        img = imread(image_path)
        result_list = DeepModel.detector.detect_faces(img)
        if len(result_list) == 1:
            [X, Y, W, H] = result_list[0]['box']
            crop = img[Y:Y+H, X:X+W]
            faces_found = DeepModel.detector.detect_faces(crop)
            if len(faces_found) == 1:
                res.append(crop)
        elif len(result_list) > 1:
            for result in result_list:
                [X, Y, W, H] = result['box']
                crop = img[Y:Y+H, X:X+W]
                faces_found = DeepModel.detector.detect_faces(crop)
                if len(faces_found) == 1:
                    res.append(crop)
        return res 
    