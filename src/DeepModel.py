import os
import pickle
from typing import Tuple
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras import models
from PIL import Image
from matplotlib.pyplot import imread
import pathlib
import zipfile
import requests
import shutil
from tqdm import tqdm
from time import perf_counter
from sklearn.neighbors import KDTree


class DeepModel(object):
    
    def __init__(
        self, 
        classifier_path:str, 
        siamese_embeddings_path:str, 
        features_path:str,
        image_train_paths:str=None,
        savory_path:str=None,
        unsavory_path:str=None,
        image_size:Tuple=(200,200)
        ) -> None:
        
        self.classifier = None
        self.image_train_paths = None
        self.features_path = features_path
        self.savory = None
        self.unsavory = None
        self.target_shape = image_size
        self.siamese_embeddings = self.__load_model(siamese_embeddings_path)
        with open(classifier_path, "rb") as f:
            self.classifier = pickle.load(f)
        with open(features_path, "rb") as f:
            self.features_path = pickle.load(f)
        with open(image_train_paths, "rb") as f:
            self.image_train_paths = pickle.load(f)
        with open(savory_path, "rb") as f:
            self.savory = pickle.load(f)
        with open(unsavory_path, "rb") as f:
            self.unsavory = pickle.load(f)
        
        dim1 = int(len(self.image_train_paths)/2)
        self.path_savory = self.image_train_paths[0:dim1]
        self.path_unsavory = self.image_train_paths[dim1:len(self.image_train_paths)]
        start = perf_counter()
        self.kdtree = KDTree(self.features_path)
        self.kdtree_s = KDTree(self.savory)
        self.kdtree_u = KDTree(self.unsavory)
        print(f"KDTree computed in: {perf_counter() - start}")


    def __load_model(self, model_path):
        path = pathlib.Path(model_path)

        if not path.exists():
            print("Downloading zipped model from Google Drive...")
            file_id = '1VfWwIVs47xKPdV5VF31iFvp0iso9ZURc'
            gdrive_link = f"https://drive.google.com/uc?export=download&confirm=9_s_&id={file_id}"
            download_file_path = pathlib.Path('./src/CNN_training/siamese.zip')

            # make an HTTP request within a context manager
            with requests.get(gdrive_link, stream=True) as r:
                
                # check header to get content length, in bytes
                total_length = int(r.headers.get("Content-Length"))
                
                # implement progress bar via tqdm
                with tqdm.wrapattr(r.raw, "read", total=total_length, desc="") as raw:
                
                    # save the output to a file
                    with open(download_file_path, 'wb') as output:
                        shutil.copyfileobj(raw, output)

            with zipfile.ZipFile(download_file_path, 'r') as fZip:
                fZip.extractall('./src/CNN_training/')

            # Delete zip file
            download_file_path.unlink()
        
        model = tf.keras.models.load_model(path)

        print(f"Model loaded from {str(path)}")

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
        if image.size != self.target_shape:
            image = image.resize(self.target_shape)
        image = np.dstack([image])
        return [self.siamese_embeddings(tf.expand_dims(image, axis=0)).numpy()[0]]
    
    
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
        feature = self.extract_features(image)
        
        start = perf_counter()
        dist_s, similar_s = self.kdtree_s.query(feature, k=k, return_distance=True)
        dist_u, similar_u = self.kdtree_u.query(feature, k=k, return_distance=True)
        print(f"{k} most similar found in: {perf_counter() - start}")
        
        return [os.path.join("..", *self.path_savory[i].parts[-4:]) for i in similar_s[0]],\
            [os.path.join("..", *self.path_unsavory[i].parts[-4:]) for i in similar_u[0]],\
            feature,\
            dist_s, dist_u 
            
            
    def cbir_performance(self, image, k:int=5) -> list:
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
        feature = self.extract_features(image)
        prediction = self.classifier.predict(feature)
        
        start = perf_counter()
        dist_s, similar_s = self.kdtree.query(feature, k=k, return_distance=True)
        print(f"{k} most similar found in: {perf_counter() - start}")
        
        return None, prediction,\
            [os.path.join("..", *self.image_train_paths[i].parts[-4:]) for i in similar_s[0]],\
            feature,\
            dist_s
    
    
    def refine_search(self, i, query_image_feature, selected_image_path:str, k:int=5):
        img = Image.open(selected_image_path)
        sel_img_emb = self.extract_features(img)
        #mean_emb = np.mean([query_image_feature, sel_img_emb], axis=0)
        mean_emb = (np.sum([query_image_feature*float(i), sel_img_emb], axis=0)) / float(i+1)
                
        dist_s, similar_s = self.kdtree_s.query(mean_emb, k=k, return_distance=True)
        dist_u, similar_u = self.kdtree_u.query(mean_emb, k=k, return_distance=True)
        
        return mean_emb,\
            [os.path.join("..", *self.path_savory[i].parts[-4:]) for i in similar_s[0]],\
            [os.path.join("..", *self.path_unsavory[i].parts[-4:]) for i in similar_u[0]],\
            dist_s, dist_u
    