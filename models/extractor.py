import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN
from torchvision.transforms.functional import to_pil_image
from torchvision.models import VGG16_Weights, ResNet50_Weights

# Pre-trained VGG16 for the feature extraction
class VGGCompressor():
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg16', weights=VGG16_Weights.DEFAULT)
        self.extractor = torch.nn.Sequential(*list(model.features.children())[:-2]).to(device)

    def extract(self, image_tensor):
        with torch.no_grad():
            features = self.extractor(image_tensor.to(self.device))
        return features.view(features.size(0), -1)
    
# Pre-trained Resnet50 for the feature extraction
class ResNetCompressor():
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', weights=ResNet50_Weights.DEFAULT)
        self.extractor = torch.nn.Sequential(*list(self.model.children())[:-2]).to(device)

    def extract(self, image_tensor):
        with torch.no_grad():
            features = self.extractor(image_tensor.to(self.device))
        return features.view(features.size(0), -1)

class FaceCropper():
    """
    A class for cropping faces from images and extracting features using a compressor model.

    Attributes:
        compressor (object): Feature extraction model.
        device (str): Device to run the model on (default is 'cpu').
        face_detector (MTCNN): Face detection model.
    """
    def __init__(self, compressor, device='cpu'):
        self.compressor = compressor
        self.device = device
        # Minimized min_face_size to detect small sized faces
        # Added margin to include the context around the detected face
        self.face_detector = MTCNN(min_face_size=2,margin=10,thresholds = [0.6, 0.6, 0.6],device = device,keep_all=True,post_process=True)
        self.face_detector.eval()

    def crop_faces(self, image_tensor):
        """
        Crop faces from the input image tensor and extract features.

        Args:
            image_tensor (torch.Tensor): Input image tensor.
            resized_img: Resize the cropped image to (200,200).
            cropped_images : Cropped image with a integrated bounding box.
            stacked_tensor: Stacked tensor of cropped images.            

        Returns:
            torch.Tensor: Extracted features using the chose compressor
        """
        cropped_images = []
        for img in image_tensor:
            pil_img = to_pil_image(img)
            boxes, _ = self.face_detector.detect(pil_img)

            if boxes is not None and len(boxes) > 0:
            # If there are bounding boxes detected
                valid_boxes = []
                for box in boxes:
                    x_min, y_min, x_max, y_max = map(int, box)
                    #prevent the bounding box is outside the image (set minimum value to 0)
                    x_min, y_min, x_max, y_max = [max(0, val) for val in [x_min, y_min, x_max, y_max]]

                    # Check if width and height are non-zero
                    if x_max > x_min and y_max > y_min:
                        valid_boxes.append((x_min, y_min, x_max, y_max))

                if len(valid_boxes) > 0:
                    # Calculate the large bounding box that encompasses all detected faces
                    x_min = min(box[0] for box in valid_boxes)
                    y_min = min(box[1] for box in valid_boxes)
                    x_max = max(box[2] for box in valid_boxes)
                    y_max = max(box[3] for box in valid_boxes)
                    
                    # Crop the image using the calculated bounding box (C,H,W)
                    cropped_img = img[:, y_min:y_max, x_min:x_max]

                    # Resize cropped image to a fixed size (e.g., 200x200) (1,C,H,W)
                    resized_img = torch.nn.functional.interpolate(cropped_img.unsqueeze(0), size=(200, 200), mode='bilinear', align_corners=False)
                    cropped_images.append(resized_img.squeeze(0))
            else:
                # If no bounding box is detected, or if only one bounding box is detected
                # In case of one bounding box, crop directly from it
                x_min, y_min, x_max, y_max = 0, 0, 200, 200
                cropped_img = img[:, y_min:y_max, x_min:x_max]
                cropped_images.append(cropped_img)
    
        # Stack the cropped images into a single tensor (B, C, H, W)
        stacked_tensor = torch.stack(cropped_images).to(self.device)     

        # Use the compressor
        extracted_features = self.compressor.extract(stacked_tensor)

        return extracted_features
