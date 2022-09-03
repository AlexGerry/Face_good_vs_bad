import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras import models
from matplotlib.pyplot import imread
from tqdm import tqdm
from time import perf_counter
from sklearn.neighbors import KDTree
from pathlib import Path
import dill


class CNN(object):
    
    def __init__(self, model_path:str=None) -> None:
        
        self.classifier = None
        self.model_path = model_path
        self.cnn = self.__load_model(self.model_path)
        self.feature_extractor = self.__get_featureExtractor(self.cnn)


    def __load_model(self, model_path):
        return models.load_model(model_path)
    
    
    #def get_best_model(self, model_path):
    #    model = None
    #    tuner = kt.BayesianOptimization(build_model,
    #                            objective=kt.Objective('val_loss', direction="min"),
    #                            directory=model_path,
    #                            max_trials = 20, overwrite=False,
    #                            project_name='tuned_model')
    #    tuner.reload()
    #    a = tuner.get_best_hyperparameters(num_trials=1)[0]
    #    model = tuner.hypermodel.build(a)
    #
    #    return model
    
    
    def __get_featureExtractor(self, model):
        extractor = models.Model(model.inputs, model.layers[-8].output)  # -3 o -5 -> 64, -8 -> 5184
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
        prediction = self.cnn.predict(img, verbose=0)
        return np.max(prediction, axis=1), np.argmax(prediction, axis=1)
    
    
    def cbir(self, image, savory_path:str=None, unsavory_path:str=None, image_train_path:str=None, k:int=5):
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
        if savory_path is None or unsavory_path is None or image_train_path is None: raise ValueError("Not a valid path!")
        # Load train bovw
        with open(savory_path, 'rb') as f: savory = dill.load(f)
        with open(unsavory_path, 'rb') as f: unsavory = dill.load(f)
        # Load train paths
        with open(image_train_path, 'rb') as f: train_paths = dill.load(f)
        
        # Divide train path
        train_paths = np.asarray(train_paths)
        path_unsavory = [x for x in train_paths if 'unsavory' in x]
        path_savory = np.setdiff1d(train_paths, path_unsavory)

        feature = self.extract_feature(image).reshape(1, -1)
        pred_score, prediction = self.predict(image)

        savory = np.reshape(savory, (len(savory),-1))
        unsavory = np.reshape(unsavory, (len(unsavory),-1))
        
        start = perf_counter()
        tree_s = KDTree(savory)
        tree_u = KDTree(unsavory)
        print(f"KDTree computed in: {perf_counter() - start}")
        
        start = perf_counter()
        dist_s, similar_s = tree_s.query(feature, k=k, return_distance=True)
        dist_u, similar_u = tree_u.query(feature, k=k, return_distance=True)
        print(f"{k} most similar found in: {perf_counter() - start}")
        
        return pred_score, prediction,\
            [os.path.join("..", *path_savory[i].split('\\')[-4:]) for i in similar_s[0]],\
            [os.path.join("..", *path_unsavory[i].split('\\')[-4:]) for i in similar_u[0]],\
            feature,\
            dist_s, dist_u  
            
            
    def cbir_performance(self, image, features_path:str=None, image_train_path:str=None, k:int=5):
        if features_path is None or image_train_path is None: raise ValueError("Not a valid path!")
        # Load train bovw
        with open(features_path, 'rb') as f: features = dill.load(f)
        # Load train paths
        with open(image_train_path, 'rb') as f: train_paths = dill.load(f)
                
        feature = self.extract_feature(image).reshape(1, -1)
        pred_score, prediction = self.predict(image)

        features = np.reshape(features, (len(features),-1))
        
        start = perf_counter()
        tree = KDTree(features)
        print(f"KDTree computed in: {perf_counter() - start}")
        
        start = perf_counter()
        dist, similar = tree.query(feature, k=k, return_distance=True)
        print(f"{k} most similar found in: {perf_counter() - start}")
        
        return pred_score, prediction,\
                [os.path.join(*train_paths[i].split('\\')[-5:]) for i in similar[0]],\
                feature,\
                dist
    
    
    def refine_search(self, i, query_image_feature, selected_image_path:str, savory_path:str=None, unsavory_path:str=None, image_train_path:str=None, k:int=5):
        if savory_path is None or unsavory_path is None or image_train_path is None: raise ValueError("Not a valid path!")
        # Load train bovw
        with open(savory_path, 'rb') as f: savory = dill.load(f)
        with open(unsavory_path, 'rb') as f: unsavory = dill.load(f)
        # Load train paths
        with open(image_train_path, 'rb') as f: train_paths = dill.load(f)
        
        # Divide train path
        train_paths = np.asarray(train_paths)
        path_unsavory = [x for x in train_paths if 'unsavory' in x]
        path_savory = np.setdiff1d(train_paths, path_unsavory)
        
        sel_img_emb = self.extract_feature(selected_image_path).reshape(1, -1)
        #mean_emb = np.mean([query_image_feature, sel_img_emb], axis=0)
        mean_emb = (np.sum([np.multiply(query_image_feature, float(i)), sel_img_emb], axis=0)) / float(i+1)
        
        savory = np.reshape(savory, (len(savory),-1))
        unsavory = np.reshape(unsavory, (len(unsavory),-1))
                
        tree_s = KDTree(savory)
        tree_u = KDTree(unsavory)
        
        dist_s, similar_s = tree_s.query(mean_emb, k=k, return_distance=True)
        dist_u, similar_u = tree_u.query(mean_emb, k=k, return_distance=True)
        
        return mean_emb,\
            [os.path.join("..", *path_savory[i].split('\\')[-4:]) for i in similar_s[0]],\
            [os.path.join("..", *path_unsavory[i].split('\\')[-4:]) for i in similar_u[0]],\
            dist_s, dist_u
            
    