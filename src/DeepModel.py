import os
import pickle
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras import models


class DeepModel(object):
    
    def __init__(self, model_path, dataset_features_path) -> None:
        self.model = models.load_model(model_path)
        
        with open(dataset_features_path, "rb") as f:
            self.features = pickle.load(f)
    
    
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
        pass
    
    
    def extract_features(self, image_path:str):
        """ This function extract a feature vector from the given image.
        
            Parameters:
            ----------
            image_path: str
                The path of the input image
                
            Returns:
            -------
            features: None
                The feature vector of the image
        
            """
        pass
    
    
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
    
    