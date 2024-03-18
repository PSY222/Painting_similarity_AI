import os
import tqdm
import pandas as pd
import urllib.request
from math import floor
from joblib import Parallel, delayed

class ImageDownloader:
        """
    Utility class for downloading and managing image data from a remote source.

    Attributes:
        loader_root (str): Root directory where image data will be stored.
        csv_remote_path (str): Remote path for CSV files.(NationalGalleryOfArt/opendata)

    Methods:
        ensure_exists(self, path, image=False): Ensure that the specified directory exists.
        get_base_dir(self): Get the base directory and create necessary subdirectories.
        thumbnail_to_local(self, base_path, object_id): Generate the local path for storing a thumbnail image.
        get_file(self, remote_url, out, timeout_seconds=10): Download a file from a remote URL.
        check_csv_exists(self, csv_name, base_dir=None): Check if a CSV file exists locally, and download it if not.
        download_painting(self, base_dir=None, percent=100): Download painting images and associated metadata.
        merge_and_filter(self, objects_df, images_df, output_file): Merge and filter DataFrames to get relevant data.
    """
    def __init__(self, loader_root):
        self.loader_root = loader_root
        self.csv_remote_path = 'https://raw.githubusercontent.com/NationalGalleryOfArt/opendata/main/data/'

    def ensure_exists(self, path, image=False):
        """Ensure that the specified directory exists."""
        if not os.path.exists(path):
            os.makedirs(path)
        elif os.listdir(path) and image:
            # Prevent downloading images in a non-empty folder
            raise OSError(f"The folder '{path}' is not empty.")

    def get_base_dir(self):
        """Get the base directory and create necessary subdirectories."""
        self.ensure_exists(self.loader_root)
        self.ensure_exists(f"{self.loader_root}/annotations")
        self.ensure_exists(f"{self.loader_root}/images",True)
        self.ensure_exists(f"{self.loader_root}/queryImg",True)
        
        return self.loader_root

    def thumbnail_to_local(self, base_path, object_id):
         """Generate the local path for storing a thumbnail image."""
        image_path = f"{base_path}/images"
        ending = f"{object_id}.jpg"
        return f"{image_path}/{ending}"
    
    def get_file(self, remote_url, out, timeout_seconds=10):
        """Download a file from a remote URL."""
        with urllib.request.urlopen(remote_url, timeout=timeout_seconds) as response:
            with open(out, "wb") as out_file:
                data = response.read()  # a `bytes` object
                out_file.write(data)
                
    def check_csv_exists(self,csv_name,base_dir=None):
        """ Check if a CSV file exists locally, and download it if not."""
        base_dir = base_dir or self.get_base_dir()
        csv_path = f"{base_dir}/annotations/{csv_name}.csv"
        if not os.path.exists(csv_name):
                self.get_file(self.csv_remote_path+f'/{csv_name}.csv', out=csv_path, timeout_seconds=100)
                print(f"{csv_name}.csv file download successful")
        return csv_path
    
    def download_painting(self, base_dir=None, percent=100):
        """
        Download painting images and associated metadata from the National Gallery of Art Open Data repository.

        Args:
            base_dir (str, optional): Base directory path. Defaults to None.
            percent (int, optional): Percentage of data to download. Defaults to 100
        """
        print("Downloading data...")
        base_dir = base_dir or self.get_base_dir()

        objects_dimensions_csv = self.check_csv_exists('objects_dimensions')
        objects_df = pd.read_csv(objects_dimensions_csv)
        
        published_images_csv = self.check_csv_exists('published_images')
        images_df = pd.read_csv(published_images_csv)

        # Merge and filter DataFrames
        painted_df = self.merge_and_filter(objects_df, images_df,base_dir)
        samples = floor(painted_df.shape[0] * (percent / 100))
        painted_df = painted_df.head(samples)

        def download_image(object_id,thumb):
            out = self.thumbnail_to_local(base_dir,object_id)
            if os.path.exists(out):
                 return
            try:
                self.get_file(thumb, out=out)
            except Exception as e:
                print(e)
                print(f"failed to get {thumb}")  

        print(f"Found {painted_df['objectid'].nunique()} images.")

        Parallel(n_jobs=16)(delayed(download_image)(object_id, thumb) for object_id, thumb in tqdm.tqdm(painted_df[['objectid', 'iiifthumburl']].values))
      
        print(f"{len(painted_df['objectid'])} images download completed")
        
    def merge_and_filter(self, objects_df, images_df, output_file):
         """
        Merge and filter DataFrames to get relevant data.

        Args:
            objects_df (DataFrame): DataFrame containing object information.
            images_df (DataFrame): DataFrame containing image URLs.
            
        Returns:
            painted_df: Filtered DataFrame containing relevant data for painting images.
                       - columns : objectid | element | iiifthumburl
        """
        painted_df = pd.merge(
            objects_df[['objectid', 'element']],
            images_df[['depictstmsobjectid', 'iiifthumburl']],
            left_on='objectid', right_on='depictstmsobjectid',
            how='inner'
        ).query("element == 'painted surface'")
        #filter 'paintings' using element column.

        painted_df = painted_df.drop_duplicates().drop('depictstmsobjectid', axis=1)
        painted_df.to_csv(output_file + '/merged.csv', index=False)

        return painted_df


if __name__ == "__main__":
    curr_path = os.getcwd()
    loader_root = curr_path + "/data"
    percent = 100  
    downloader = ImageDownloader(loader_root)
    downloader.download_painting(percent=percent)
