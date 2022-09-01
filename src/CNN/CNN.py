import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras import models
from keras.preprocessing import image
from PIL import Image
from mtcnn.mtcnn import MTCNN
from matplotlib.pyplot import imread
import keras_tuner as kt
from utils import build_model
from tqdm import tqdm
from time import perf_counter
from sklearn.neighbors import KDTree
from pathlib import Path
import dill
import sys


class CNN(object):
    detector = MTCNN()
    
    def __init__(self, model_path:str=None) -> None:
        
        self.classifier = None
        self.model_path = model_path
        self.cnn = self.__load_model(self.model_path)
        self.feature_extractor = self.__get_featureExtractor(self.cnn)


    def __load_model(self, model_path):
        return models.load_model(model_path)
    
    
    def get_best_model(self, model_path):
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
    
    
    def __get_featureExtractor(self, model):
        extractor = models.Model(model.inputs, model.layers[-8].output) # Dense(64,...)
        return extractor
    
    
    def load_image(self, image_path):
        img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
        input_arr = tf.keras.utils.img_to_array(img) / 255.
        input_arr = np.array([input_arr])  # Convert single image to a batch.
        return input_arr
    
    
    def extract_feature(self, image_path):
        """ This function extract a feature vector from the given image path.
        
            Parameters:
            ----------
            image: str
                The input image path
                
            Returns:
            -------
            features: None
                The feature vector of the image
        
            """
        img = self.load_image(image_path)
        return self.feature_extractor.predict(img, verbose=0)[0]
    
    
    def extract_from_folder(self, image_folder_path:str, image_format:str='jpeg', save_path:str=None):
        if save_path is None: raise ValueError("Save path is none!")
        os.makedirs(save_path, exist_ok=True)
        
        paths = list(Path(image_folder_path).rglob(f"*.{image_format}"))
        train_features, train_labels, train_path = [], [], []
        
        for image_path in (pbar := tqdm(paths)):
            pbar.set_description(f"Extracting CNN features from image {image_path}...")
            feat = self.extract_feature(image_path)
            train_features.append(feat)
            label = image_path.parts[-2]
            train_labels.append(label)
            train_path.append(str(image_path))
            
        try:
            with open(f'{save_path}/cnn_train_features.pkl', 'wb') as f: dill.dump(train_features, f)
            with open(f'{save_path}/cnn_train_paths.pkl', 'wb') as f: dill.dump(train_path, f)
            with open(f'{save_path}/cnn_train_labels.pkl', 'wb') as f: dill.dump(train_labels, f)
        except Exception as e:
            print(f"{e}: Error during saving...")
        
        return train_features, train_labels, train_path

    
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
        img = self.load_image(image_path)
        return np.argmax(self.cnn.predict(img, verbose=0), axis=1)
    
    
    def cbir(self, image, k:int=5):
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
    
    
    def save_model(self, path):
        with open(path, 'wb') as f: dill.dump(self, f)
        
    
    @staticmethod
    def load_model(path):
        sys.path.insert(0, "./src")
        with open(path, 'rb') as f: model = dill.load(f)    
        return model