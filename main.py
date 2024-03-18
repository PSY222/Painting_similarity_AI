import os
import argparse
import torch
from utils.download_img import ImageDownloader
from utils.evaluation import evaluation_metrics
from utils.retrieve import ImageRetrieval
from models.vgg_compressor import VGGCompressor
from models.resnet_compressor import ResNetCompressor

def main(args):
    # Set up basic parameters
    curr_path = os.getcwd()
    loader_root = os.path.join(curr_path, "data")
    percent = args.percent
    data_path = os.path.join(curr_path, "data/images")
    query_image_path = args.query_image_path

    # Initialize compressor and device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.compressor == 'vgg':
        compressor = VGGCompressor(device=device)
    elif args.compressor == 'resnet':
        compressor = ResNetCompressor(device=device)
    else:
        raise ValueError("Check the compressor name again.")

    # Download paintings
    if args.download:
        downloader = ImageDownloader(loader_root)
        downloader.download_painting(percent=percent)

    # Retrieve similar images
    image_retrieval = ImageRetrieval(compressor=compressor, img_path=data_path,k=args.k device=device)
    similar_images = image_retrieval.retrieve_similar_images(query_image_path, metric=args.metric, face_crop=args.face_crop)
    
    # Evaluation metrics
    # 1. Visualisation
    eval = evaluation_metrics(data_path)
    eval.visualize_images(similar_images,query_image_path)
    
    # 2. Quantitative Evaluation
    eval.calculate_average_precision(similar_images,query_image_path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve similar images.")
    parser.add_argument('--compressor', type=str, default='vgg', choices=['vgg', 'resnet'],
                        help="Choose the compressor model: 'vgg' or 'resnet'.(default: vgg).")
    parser.add_argument('--download', action='store_true', help="Download paintings if specified.")
    parser.add_argument('--k', type=int, default=5, help="Number of similar images to retrieve")
    parser.add_argument('--percent', type=int, default=100, help="Percentage of paintings to download. Default is 100.")
    parser.add_argument('--query_image_path', type=str, ,default="./data/images/0.jpg",
                        help="Path to the query image.")
    parser.add_argument('--metric', type=str, default='euclidean', choices=['euclidean', 'cosine','manhattan'],
                        help="Distance metric for the similarity calculation. (default: euclidean).")
    parser.add_argument('--face_crop', action='store_true', help="Enable face cropping to focus on facial similarity.")

    args = parser.parse_args()
    main(args)
