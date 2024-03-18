import matplotlib.pyplot as plt
from PIL import Image

class evaluation_metrics:
    def __init__(self,img_path):
        self.img_path img_path
        
    def visualize_images(self, images, query_image_path):
        """
        Visualizes the query image along with its similar images.

        Args:
            images (list): List of tuples containing image paths and their distances.
            query_image_path (str): Path to the query image.
        """
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
            axes[i].set_title(f'Similar Image {i} ({file_name})')
            axes[i].axis('off')
        plt.tight_layout()
        plt.show()

    
    import numpy as np
    from skimage.metrics import structural_similarity as ssim
    from PIL import Image

    def calculate_average_precision(query_image_path, similar_images_paths):
        # Load the query image
        query_image = np.array(Image.open(query_image_path).convert("RGB"))

        # Calculate SSIM for each similar image
        ssim_scores = []
        for image_path in similar_images_paths:
            similar_image = np.array(Image.open(image_path).convert("RGB"))
            ssim_score = ssim(query_image, similar_image, multichannel=True)
            ssim_scores.append(ssim_score)

        # Sort the similar images by SSIM scores
        sorted_indices = np.argsort(ssim_scores)[::-1]
        sorted_similar_images_paths = [similar_images_paths[i] for i in sorted_indices]

        # Calculate Average Precision (AP)
        precision = []
        num_retrieved_images = len(similar_images_paths)
        num_relevant_retrieved = 0
        for i, idx in enumerate(sorted_indices):
            if idx == 0:  # Assuming the first retrieved image is relevant (query image)
                num_relevant_retrieved += 1
                precision_at_i = num_relevant_retrieved / (i + 1)
                precision.append(precision_at_i)

        average_precision = sum(precision) / len(precision) if precision else 0.0
        
        print('AP(SSIM score) : ', average_precision)

        return average_precision

# Example usage:
# query_image_path = "query_image.jpg"
# similar_images_paths = ["similar_image1.jpg", "similar_image2.jpg", "similar_image3.jpg"]
# ap = calculate_average_precision(query_image_path, similar_images_paths)
# print("Average Precision:", ap)

def average_precision(found: Sequence[str], ground_truth: str) -> float:
    groups = list(map(lambda index: (found[index], found[: index + 1]), range(len(found))))
    groups = list(filter(lambda group: group[0] == ground_truth, groups))
    precisions = list(map(lambda group: precision(found=group[1], ground_truth=group[0]), groups))
    return fmean(precisions) if precisions else 0.


def mean_average_precision(retrievals: Sequence[Sequence[str]], labels: Sequence[str]) -> float:
    average_precisions = list(
        map(lambda groups: average_precision(found=groups[0], ground_truth=groups[1]), list(zip(retrievals, labels)))
    )