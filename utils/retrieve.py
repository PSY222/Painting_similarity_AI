import os
import torch
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils.data_loader import ImageDataset
from sklearn.neighbors import NearestNeighbors
from torchvision.transforms import transforms
from models.extractor import VGGCompressor, ResNetCompressor, FaceCropper


class ImageRetrieval:
    """
    A class for retrieving similar images based on feature similarity using a given compressor model.

    Attributes:
        compressor (object): Feature extraction model.
        img_path (str): Path to the directory containing images.
        k (int): Number of similar images to retrieve (default is 5).
        face_crop (bool): Flag to indicate whether to perform face cropping during feature extraction (default is False).
        device (str): Device to run the model on (default is 'cpu').
        merged_df (pd.DataFrame): DataFrame containing metadata of merged images.
        features (list): List of extracted features.
        image_paths (list): List of image paths corresponding to the extracted features.
    """
    def __init__(self, compressor, img_path, k=5, face_crop=False, device='cpu'):  
        self.compressor = compressor
        self.compressor.device = device
        self.merged_df = pd.read_csv('./data/merged.csv')
        self.img_path = img_path
        self.device = device
        self.k = k     
        self.features, self.image_paths = self.extract_features(face_crop)

    def extract_features(self,face_crop=False):
        """
        Extracts features from the images.
        """
        #Load the images
        image_dataset = ImageDataset(self.merged_df)
        data_loader = DataLoader(image_dataset, batch_size=32, shuffle=False)

        features = []
        image_paths = []
        
        if face_crop:
            face_cropper = FaceCropper(self.compressor,device=self.device)
                
        print('Building Feature Vector List')
        for image_tensors in tqdm(data_loader, total=len(data_loader)):
            # If face_crop is set to True, crop the image with detected faces' bounding box.
            if face_crop:
                extracted_features = face_cropper.crop_faces(image_tensors)
            else:
                extracted_features = self.compressor.extract(image_tensors.to(self.device))
        
            extracted_features = extracted_features.view(extracted_features.size(0), -1).cpu().numpy()
            features.extend(extracted_features)
            image_paths.extend(image_dataset.dataFrame.iloc[len(features) - len(extracted_features):len(features)]['objectid'].tolist())
        return features, image_paths
    
    def retrieve_similar_images(self, query_image_path, metric='cosine',face_crop=False):
        """
        Retrieves similar images to the query image.

        Args:
            query_image_path (str): Path to the query image.
            metric (str, optional): Distance metric to be used (default -> 'cosine').
            face_crop (bool, optional): Whether to perform face cropping during feature extraction (default is False).

        Returns:
            list: List of tuples containing image paths and their distances.
        """
        # Pre-process query image
        self.transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Resize((200,200)),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]) 
        image = Image.open(query_image_path).convert("RGB")
        query_image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        if face_crop:
            face_cropper = FaceCropper(self.compressor,device=self.device)
            query_features = face_cropper.crop_faces(query_image_tensor)
            # No faces detected, extract feature from the original query image tensor
            if query_features is None:
                    query_features = self.compressor.extract(query_image_tensor)
        else:
            query_features = self.compressor.extract(query_image_tensor)
            
        # Exclude query image to avoid self-similarity in search.    
        query_object_id = os.path.basename(query_image_path).split(".")[0]
        query_index = self.merged_df[self.merged_df['objectid'] == int(query_object_id)].index.tolist()[0]
        except_query = np.delete(self.features, query_index+1, axis=0) 
        
        # Train KNN
        knn = NearestNeighbors(n_neighbors=self.k, metric=metric)
        knn.fit(except_query)
        
        # Find similar images from the extracted query_features from the quert_image
        query_features = query_features.cpu().numpy() 
        distances, indices = knn.kneighbors(query_features)
        similar_images = [(self.image_paths[i], distances[0, j]) for j, i in enumerate(indices[0])]
        similar_images = similar_images[1:]  # Skip the first element (query image)
        similar_images.sort(key=lambda x: x[1])  # Sort the similar images by distance
        
        return similar_images



