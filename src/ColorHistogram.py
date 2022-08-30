from pathlib import Path
import dill
import cv2
from tqdm import tqdm
from time import perf_counter
import os, sys
from sklearn.svm import SVC
from sklearn.neighbors import KDTree, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


class ColorHistogram(object):
    __name__ = "ColorHistogram"
    
    def __init__(self, num_bins:int=8) -> None:
        self.num_bins = (num_bins, num_bins, num_bins)
        self.model = SVC(random_state=42)
        self.knn = KNeighborsClassifier()
        self.forest = RandomForestClassifier(random_state=42)
        
    
    def compute_histogram(self, image):
        hist = cv2.calcHist([image], [0, 1, 2], None, self.num_bins,
                    [0, 256, 0, 256, 0, 256])
        norm_hist = cv2.normalize(hist, hist).flatten()
        return norm_hist
    
    
    def extract_descriptor(self, data_path:str, image_format:str='jpeg'):
        paths = list(Path(data_path).rglob(f"*.{image_format}"))
        train_features, train_labels, train_path = [], [], []
        
        for image_path in (pbar := tqdm(paths)):
            pbar.set_description(f"Extracting Color Histogram from image {image_path}...")
            im = cv2.imread(str(image_path))
            norm_hist = self.compute_histogram(im)
            train_features.append(norm_hist)
            label = image_path.parts[-2]
            train_labels.append(label)
            train_path.append(str(image_path))
            
        return train_features, train_labels, train_path
    
    
    def create_train_descriptors(self, image_folder_path:str, image_format:str='jpeg', save_path:str=None):
        if save_path is None: raise ValueError("Save path is none!")
        os.makedirs(save_path, exist_ok=True)
        
        descr, labels, train_paths = self.extract_descriptor(image_folder_path, image_format)
        
        start = perf_counter()
        self.model.fit(descr, labels)
        print(f"SVC fitted in: {perf_counter() - start}")
        
        start = perf_counter()
        self.knn.fit(descr, labels)
        print(f"KNN fitted in: {perf_counter() - start}")
        
        start = perf_counter()
        self.forest.fit(descr, labels)
        print(f"Random Forest fitted in: {perf_counter() - start}")
        
        try:
            with open(f'{save_path}/train_color_histogram.pkl', 'wb') as f: dill.dump(descr, f)
            with open(f'{save_path}/train_paths.pkl', 'wb') as f: dill.dump(train_paths, f)
            with open(f'{save_path}/train_labels.pkl', 'wb') as f: dill.dump(labels, f)
            self.save_model(save_path+"/histogram_model.pkl")
        except Exception as e:
            print(f"{e}: Error during saving...")
        
        return descr, labels, train_paths
    
    
    def predict_image(self, image_path):        
        im = cv2.imread(str(image_path))
        # Compute histogram
        hist = self.compute_histogram(im)
        # Predict
        return self.forest.predict([hist]), [hist]
    
    
    @staticmethod
    def cbir(model_path, image_path, savory_path:str=None, unsavory_path:str=None, image_train_path:str=None, k:int=5   ):
        if savory_path is None or unsavory_path is None or model_path is None or image_train_path is None: raise ValueError("Not a valid path!")
        # Load train bovw
        with open(savory_path, 'rb') as f: savory = dill.load(f)
        with open(unsavory_path, 'rb') as f: unsavory = dill.load(f)
        # Load train paths
        with open(image_train_path, 'rb') as f: train_paths = dill.load(f)
        # Load model
        color_model = ColorHistogram.load_model(model_path)
        # Divide train path
        dim1 = int(len(train_paths)/2)
        path_savory = train_paths[0:dim1]
        path_unsavory = train_paths[dim1:len(train_paths)]
        
        prediction, feature = color_model.predict_image(image_path)
        start = perf_counter()
        tree_s = KDTree(savory)
        tree_u = KDTree(unsavory)
        print(f"KDTree computed in: {perf_counter() - start}")
        
        start = perf_counter()
        similar_s = tree_s.query(feature, k=k, return_distance=False)
        similar_u = tree_u.query(feature, k=k, return_distance=False)
        print(f"{k} most similar found in: {perf_counter() - start}")
        
        return prediction, [os.path.join(*path_savory[i].split('\\')[-5:]) for i in similar_s[0]], [os.path.join(*path_unsavory[i].split('\\')[-5:]) for i in similar_u[0]]
    
    
    def save_model(self, path):
        with open(path, 'wb') as f: dill.dump(self, f)
        
    
    @staticmethod
    def load_model(path):
        sys.path.insert(0, "./src")
        with open(path, 'rb') as f: model = dill.load(f)    
        return model