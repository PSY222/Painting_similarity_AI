import torch
from torchvision import transforms
from torchvision.models import VGG16_Weights, ResNet50_Weights
from PIL import Image
import numpy as np


class VGGCompressor():
    def __init__(self):
        super().__init__()
        model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg16', weights=VGG16_Weights.DEFAULT)
        self.extractor = torch.nn.Sequential(*list(model.features.children())[:-2])

    def extract(self, image_tensor):
        with torch.no_grad():
            features = self.extractor(image_tensor)
        return features.view(features.size(0), -1)
    
class ResNetCompressor():
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', weights=ResNet50_Weights.DEFAULT)
        self.extractor = torch.nn.Sequential(*list(self.model.children())[:-2])

    def extract(self, image_tensor):
        # image_tensor.shape = [32,3,200,200]
        with torch.no_grad():
            features = self.extractor(image_tensor)
            # features.shape = [32,2048,7,7]
        return features.view(features.size(0), -1)
