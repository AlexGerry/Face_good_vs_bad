import numpy as np
import dill
from sklearn.svm import SVC
from sklearn.neighbors import KDTree, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from time import perf_counter
import os, sys


class CombinedModel(object):
    
    def __init__(self, features_path:list, labels_path:str, images_path:str) -> None:
        self.features = []
        for path in features_path:
            with open(path, 'rb') as f: self.features.append(dill.load(f))
        self.features = np.concatenate(self.features, axis=1)
        
        with open(labels_path, 'rb') as f: self.labels = dill.load(f)
        with open(images_path, 'rb') as f: self.images_path = dill.load(f)
        
        self.model = SVC(random_state=42)
        self.knn = KNeighborsClassifier()
        self.forest = RandomForestClassifier(random_state=42)
        
        self.models = []
        
        
    def fit_model(self, save_path:str):
        start = perf_counter()
        self.model.fit(self.features, self.labels)
        print(f"SVC fitted in: {perf_counter() - start}")
        
        start = perf_counter()
        self.knn.fit(self.features, self.labels)
        print(f"KNN fitted in: {perf_counter() - start}")
        
        start = perf_counter()
        self.forest.fit(self.features, self.labels)
        print(f"Random Forest fitted in: {perf_counter() - start}")
        
        os.makedirs(save_path, exist_ok=True)
        self.save_model(save_path+"/combined_model.pkl")
        
        
    def get_models(self, *models:object):
        for m in models:
            self.models.append(m)
        
        
    def extract_descriptor(self, data_path:str, image_format:str='jpeg'):
        if len(self.models) == 0 : raise Exception("You must first upload some models with get_models function!") 
        train_features, train_labels, train_path = [], [], []
        
        for m in self.models:
            ff = []
            ff, train_labels, train_path = getattr(m, "extract_descriptor")(data_path, image_format)
            train_features.append(ff)
            
        train_features = np.concatenate(train_features, axis=1)
            
        return train_features, train_labels, train_path
    
    
    def save_model(self, path):
        with open(path, 'wb') as f: dill.dump(self, f)
        
    
    @staticmethod
    def load_model(path):
        sys.path.insert(0, "./src")
        with open(path, 'rb') as f: model = dill.load(f)    
        return model
    
    
    def predict_image(self, image_path): 
        if len(self.models) == 0 : raise Exception("You must first upload some models with get_models function!")        
        descr = []
        for m in self.models:
            ff = []
            _, ff = getattr(m, "predict_image")(image_path)
            descr.append(ff)
        descr = np.concatenate(descr, axis=1)
        # Predict
        return self.forest.predict(descr), descr
    
    
    def cbir(model_path, image_path, features_train_path:str=None, image_train_path:str=None, k:int=10):
        if model_path is None: raise ValueError("Not a valid path!")
        # Load model
        comb_model = CombinedModel.load_model(model_path)
        comb_model.get_models(CombinedModel.load_model('./src/BOVW/bovw/bovw.pkl'), CombinedModel.load_model('./src/Color/color/histogram_model.pkl'))
        
        prediction, feature = comb_model.predict_image(image_path)
        start = perf_counter()
        tree = KDTree(comb_model.features)
        print(f"KDTree computed in: {perf_counter() - start}")
        
        start = perf_counter()
        similar = tree.query(feature, k=k, return_distance=False)
        print(f"{k} most similar found in: {perf_counter() - start}")
        
        return prediction, [os.path.join(*comb_model.images_path[i].split('\\')[-5:]) for i in similar[0]]