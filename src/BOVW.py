import cv2
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy.cluster.vq import vq
from time import perf_counter
import dill
from sklearn.neighbors import KDTree
import os
import sys
from matplotlib import pyplot as plt


class BOVW(object):
    
    def __init__(self, num_cluster:int=100, step_size:int=15, img_dim:int=300) -> None:
        self.num_clusters = num_cluster
        self.kmeans = KMeans(n_clusters = self.num_clusters, random_state = 42)
        self.model = SVC(random_state=42, probability=True)
        self.step_size = step_size
        self.img_dim = img_dim
        
    
    def extract_Sifts(self, data_path:str, image_format:str='jpeg'):
        paths = list(Path(data_path).rglob(f"*.{image_format}"))
        train_features, train_labels, train_path = [], [], []
        
        method = cv2.SIFT_create()
        kp = [cv2.KeyPoint(x, y, self.step_size) for y in range(0, self.img_dim, self.step_size) 
                                    for x in range(0, self.img_dim, self.step_size)]
        
        for image_path in (pbar := tqdm(paths)):
            pbar.set_description(f"Extracting SIFT from image {image_path}...")
            im = cv2.imread(str(image_path), 0)
            im = cv2.resize(im, (self.img_dim, self.img_dim))
            _, descr = method.compute(im, kp)
            label = image_path.parts[-2]
            train_features.append(descr)
            train_labels.append(label)
            train_path.append(str(image_path))
                    
        return train_features, train_labels, train_path
    
    
    def create_bovw(self, features, labels):
        descr = np.vstack(features)  
        start = perf_counter()
        self.kmeans.fit(descr)
        print(f"Visual words computed in: {perf_counter() - start}")   
        return self.compute_histogram(features, labels)
    
    
    def compute_histogram(self, features, labels):
        start = perf_counter()
        im_features = np.zeros((len(labels), self.num_clusters), "float32")
        visual_words = self.kmeans.cluster_centers_
        for i in range(len(labels)):
            words, _ = vq(features[i], visual_words)
            for w in words:
                im_features[i][w] += 1
            im_features[i] /= np.sum(im_features[i])
        print(f"BOVW computed in: {perf_counter() - start}")   
        return im_features
    
    
    def create_train_Vocabulary(self, image_folder_path:str, image_format:str='jpeg', save_path:str=None):
        if save_path is None: raise ValueError("Save path is none!")
        os.makedirs(save_path, exist_ok=True)
        
        descr, labels, train_paths = self.extract_Sifts(image_folder_path, image_format)
        train_bovw = self.create_bovw(descr, labels)
        start = perf_counter()
        self.model.fit(train_bovw, labels)
        print(f"SVC fitted in: {perf_counter() - start}")
        
        try:
            with open(f'{save_path}/train_bovw.pkl', 'wb') as f: dill.dump(train_bovw, f)
            with open(f'{save_path}/train_paths.pkl', 'wb') as f: dill.dump(train_paths, f)
            with open(f'{save_path}/train_labels.pkl', 'wb') as f: dill.dump(labels, f)
            self.save_model(save_path+"/bovw.pkl")
        except Exception as e:
            print(f"{e}: Error during saving...")
        return train_bovw, labels, train_paths
    
    
    def predict_image(self, image_path):
        # Extract sift
        sifts = []
        method = cv2.SIFT_create()
        kp = [cv2.KeyPoint(x, y, self.step_size) for y in range(0, self.img_dim, self.step_size) 
                                    for x in range(0, self.img_dim, self.step_size)]
        
        im = cv2.imread(str(image_path), 0)
        im = cv2.resize(im, (self.img_dim, self.img_dim))
        _, descr = method.compute(im, kp)
        sifts.append(descr)
        # Compute histogram
        bovw_hist = self.compute_histogram(sifts, [1])
        # Predict
        return self.model.predict_proba(bovw_hist), self.model.predict(bovw_hist), bovw_hist
    
    
    @staticmethod
    def cbir(bovw_path, image_path, savory_path:str=None, unsavory_path:str=None, image_train_path:str=None, k:int=5):
        if savory_path is None or unsavory_path is None or bovw_path is None or image_train_path is None: raise ValueError("Not a valid path!")
        # Load train bovw
        with open(savory_path, 'rb') as f: savory = dill.load(f)
        with open(unsavory_path, 'rb') as f: unsavory = dill.load(f)
        # Load train paths
        with open(image_train_path, 'rb') as f: train_paths = dill.load(f)
        # Load model
        bovw = BOVW.load_model(bovw_path)
        
        # Divide paths
        train_paths = np.asarray(train_paths)
        path_unsavory = [x for x in train_paths if 'unsavory' in x]
        path_savory = np.setdiff1d(train_paths, path_unsavory)
        
        pred_score, prediction, feature = bovw.predict_image(image_path)
        start = perf_counter()
        tree_s = KDTree(savory)
        tree_u = KDTree(unsavory)
        print(f"KDTree computed in: {perf_counter() - start}")
        
        start = perf_counter()
        dist_s, similar_s = tree_s.query(feature, k=k, return_distance=True)
        dist_u, similar_u = tree_u.query(feature, k=k, return_distance=True)
        print(f"{k} most similar found in: {perf_counter() - start}")
        
        return pred_score, prediction,\
            [os.path.join(*path_savory[i].split('\\')[-5:]) for i in similar_s[0]],\
            [os.path.join(*path_unsavory[i].split('\\')[-5:]) for i in similar_u[0]],\
            feature,\
            dist_s, dist_u
            
    
    @staticmethod
    def cbir_performance(model_path, image_path, features_path:str=None, image_train_path:str=None, k:int=5):
        if features_path is None or model_path is None or image_train_path is None: raise ValueError("Not a valid path!")
        # Load train bovw
        with open(features_path, 'rb') as f: features = dill.load(f)
        # Load train paths
        with open(image_train_path, 'rb') as f: train_paths = dill.load(f)
        # Load model
        bovw = BOVW.load_model(model_path)
                
        pred_score, prediction, feature = bovw.predict_image(image_path)
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
    
    
    def save_model(self, path):
        with open(path, 'wb') as f: dill.dump(self, f)
        

    @staticmethod
    def load_model(path):
        sys.path.insert(0, "./src")
        with open(path, 'rb') as f: model = dill.load(f)    
        return model
    
    
    @staticmethod
    def show_results(model, X, y, imgs, show_positive=True):
        preds = model.predict(X)
        idxs = np.where((preds == y) == show_positive)[0]
        
        fig = plt.figure(figsize=(20, 10))
        idx = np.random.permutation(idxs)
        for i in range(8):
            try:
                plt.subplot(2, 4, i + 1)
                val = idx[i]
                plt.title(f"Pred={preds[val]}, Label={y[val]}")
                plt.axis('off')
                plt.imshow(plt.imread(imgs[val]))
            except Exception as e:
                pass
        plt.show()
        
    
    @staticmethod
    def refine_search(i, query_image_feature, selected_image_path:str, bovw_path:str, savory_path:str=None, unsavory_path:str=None, image_train_path:str=None, k:int=5):
        if savory_path is None or unsavory_path is None or bovw_path is None or image_train_path is None: raise ValueError("Not a valid path!")
        # Load train bovw
        with open(savory_path, 'rb') as f: savory = dill.load(f)
        with open(unsavory_path, 'rb') as f: unsavory = dill.load(f)
        # Load train paths
        with open(image_train_path, 'rb') as f: train_paths = dill.load(f)
        # Load model
        bovw = BOVW.load_model(bovw_path)
        
        # Divide paths
        train_paths = np.asarray(train_paths)
        path_unsavory = [x for x in train_paths if 'unsavory' in x]
        path_savory = np.setdiff1d(train_paths, path_unsavory)
        
        _, _, sel_img_emb = bovw.predict_image(os.path.abspath(selected_image_path))
        #mean_emb = np.mean([query_image_feature, sel_img_emb], axis=0)
        mean_emb = (np.sum([np.multiply(query_image_feature, float(i)), sel_img_emb], axis=0)) / float(i+1)
        
        tree_s = KDTree(savory)
        tree_u = KDTree(unsavory)
        
        dist_s, similar_s = tree_s.query(mean_emb, k=k, return_distance=True)
        dist_u, similar_u = tree_u.query(mean_emb, k=k, return_distance=True)
        
        return mean_emb,\
            [os.path.join(*path_savory[i].split('\\')[-5:]) for i in similar_s[0]],\
            [os.path.join(*path_unsavory[i].split('\\')[-5:]) for i in similar_u[0]],\
            dist_s, dist_u