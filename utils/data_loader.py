import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    """
    Custom PyTorch Dataset class for loading images from a DataFrame.

    Args:
        dataFrame (DataFrame): DataFrame containing image information.(merged.csv)
        image_dir (str): Directory path where images are stored.
        transform (callable, optional): Optional transform to be applied to the images.

    Methods:
        __getitem__(self, idx): Retrieves and transforms the image at the specified index.
        __len__(self): Returns the total number of images in the dataset.
    """
        
    def __init__(self, dataFrame, image_dir="./data/images", transform=None):
        self.dataFrame = dataFrame
        self.image_dir = image_dir
        self.transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Resize((200,200)),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, str(self.dataFrame.iloc[idx]['objectid'])+'.jpg')
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image
    
    def __len__(self):
        return len(self.dataFrame)

