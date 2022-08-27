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
import pathlib
import zipfile
import requests
import shutil
from tqdm import tqdm


class DeepModel(object):
    detector = MTCNN()
    
    def __init__(self, classifier_path:str, siamese_embeddings_path:str, image_size:Tuple=(200,200)) -> None:
        self.classifier = None
        self.siamese_embeddings = self.__load_model(siamese_embeddings_path)
        self.target_shape = image_size
        with open(classifier_path, "rb") as f:
            self.classifier = pickle.load(f)

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
    