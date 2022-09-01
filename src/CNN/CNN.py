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
import keras_tuner as kt
from utils import build_model
from tqdm import tqdm
from time import perf_counter
from sklearn.neighbors import KDTree


class CNN(object):
    detector = MTCNN()
    
    def __init__(self, best_model_path:str=None) -> None:
        
        self.classifier = None
        self.best_model_path = best_model_path
        self.cnn = self.__load_model(self.best_model_path)
        
       
        #start = perf_counter()
        #self.kdtree_s = KDTree(self.savory)
        #self.kdtree_u = KDTree(self.unsavory)
        #print(f"KDTree computed in: {perf_counter() - start}")


    def __load_model(self, model_path):
        model = None
        tuner = kt.BayesianOptimization(build_model,
                                objective=kt.Objective('val_loss', direction="min"),
                                directory=model_path,
                                max_trials = 20, overwrite=False,
                                project_name='tuned_model')
        tuner.reload()
        a = tuner.get_best_hyperparameters(num_trials=1)[0]
        model = tuner.hypermodel.build(a)

        return model

    
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
        
        return 
    
    
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
        
        return 
    
    
    def cbir(self, image, k:int=5) -> list:
        """ This function outputs the most k similar image to the one given in input.
        
            Parameters:
            ----------
            image: Any
                The input image
            k: int
                The number of images to retrieve
                
            Returns:
            -------
            most_similar: list
                The k most similar images path
        
            """
        
        
        return 
    
    
    def refine_search(self, query_image_feature, selected_image_path:str, k:int=5):
       
        
        return 
    
    
    @staticmethod
    def find_faces(image_path):
        res = []
        img = imread(image_path)
        result_list = CNN.detector.detect_faces(img)
        if len(result_list) == 1:
            [X, Y, W, H] = result_list[0]['box']
            crop = img[Y:Y+H, X:X+W]
            faces_found = CNN.detector.detect_faces(crop)
            if len(faces_found) == 1:
                res.append(crop)
        elif len(result_list) > 1:
            for result in result_list:
                [X, Y, W, H] = result['box']
                crop = img[Y:Y+H, X:X+W]
                faces_found = CNN.detector.detect_faces(crop)
                if len(faces_found) == 1:
                    res.append(crop)
        return res 
    