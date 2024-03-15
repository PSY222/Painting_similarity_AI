import os
import torch
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from torchvision.transforms import transforms
from utils.data_loader import ImageDataset
from torch.utils.data import DataLoader
from models.extractor import VGGCompressor, ResNetCompressor  


class ImageRetrieval:
    def __init__(self, compressor, img_path, k=5):  # top k
        self.compressor = compressor
        self.img_path = img_path
        self.k = k     
        self.features, self.image_paths = self.extract_features()
    
    def visualize_images(self, images, query_image_path):
        # Load query image
        query_image = Image.open(query_image_path)

        # Plot query image
        fig, axes = plt.subplots(1, len(images) + 1, figsize=(15, 5))
        axes[0].imshow(query_image)
        axes[0].set_title("Query Image")
        axes[0].axis('off')

        # Plot similar images
        for i, (file_name, distance) in enumerate(images, start=1):
            image = Image.open(os.path.join(self.img_path, str(file_name)+'.jpg'))
            axes[i].imshow(image)
            axes[i].set_title(f'Similar Image {i}\nDistance: {distance:.2f}')
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    def extract_features(self):
        merged_df = pd.read_csv('./data/merged.csv')
        merged_df = merged_df[merged_df['objectid'] != 10] 

        image_dataset = ImageDataset(merged_df)
        data_loader = DataLoader(image_dataset, batch_size=32, shuffle=False)

        features = []
        image_paths = []

        print('Building feature vector list')
        for image_tensors in tqdm(data_loader, total=len(data_loader)):
            extracted_features = self.compressor.extract(image_tensors)
            extracted_features = extracted_features.view(extracted_features.size(0), -1).cpu().numpy()
            features.extend(extracted_features)
            image_paths.extend(image_dataset.dataFrame.iloc[len(features) - len(extracted_features):len(features)]['objectid'].tolist())

        return features, image_paths

    def retrieve_similar_images(self, query_image_path, metric='euclidean'):
        self.transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Resize((200,200)),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]) 
        image = Image.open(query_image_path).convert("RGB")
        query_image_tensor = self.transform(image).unsqueeze(0)

        query_features = self.compressor.extract(query_image_tensor)
        query_features_flat = query_features.view(query_features.size(0), -1).cpu().numpy()

        knn = NearestNeighbors(n_neighbors=self.k, metric=metric)
        knn.fit(self.features)

        distances, indices = knn.kneighbors(query_features)
        similar_images = [(self.image_paths[i], distances[0, j]) for j, i in enumerate(indices[0])]
        return similar_images


if __name__ == "__main__":
    # Example usage with VGGCompressor
    vgg_compressor = vgg()
    data_path = "../data/images"
    image_retrieval = ImageRetrieval(compressor=vgg, data_path=data_path)
    query_image_path = "../data/images/0.jpg"

    similar_images = image_retrieval.retrieve_similar_images(query_image_path)
    for image_path, distance in similar_images:
        print(f"Image: {image_path}, Distance: {distance}")
