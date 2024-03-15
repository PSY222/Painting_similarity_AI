import os
import tqdm
import pandas as pd
import urllib.request
from math import floor
from joblib import Parallel, delayed

class ImageDownloader:
    def __init__(self, loader_root):
        self.loader_root = loader_root
        # self.failed_object_ids = []
        self.csv_remote_path = 'https://raw.githubusercontent.com/NationalGalleryOfArt/opendata/main/data/'

    def ensure_exists(self, path,image=False):
        if not os.path.exists(path):
            os.makedirs(path)
        elif os.listdir(path) and image:
            raise OSError(f"The folder '{path}' is not empty.")

    def get_base_dir(self):
        self.ensure_exists(self.loader_root)
        self.ensure_exists(f"{self.loader_root}/annotations")
        self.ensure_exists(f"{self.loader_root}/images",True)
        return self.loader_root

    def thumbnail_to_local(self, base_path, object_id):
        image_path = f"{base_path}/images"
        ending = f"{object_id}.jpg"
        return f"{image_path}/{ending}"
    
    def get_file(self, remote_url, out, timeout_seconds=10):
            with urllib.request.urlopen(remote_url, timeout=timeout_seconds) as response:
                with open(out, "wb") as out_file:
                    data = response.read()  # a `bytes` object
                    out_file.write(data)
                
    def check_file_exists(self,csv_name,base_dir=None):
        base_dir = base_dir or self.get_base_dir()
        csv_path = f"{base_dir}/annotations/{csv_name}.csv"
        if not os.path.exists(csv_name):
                self.get_file(self.csv_remote_path+f'/{csv_name}.csv', out=csv_path, timeout_seconds=100)
                print(f"{csv_name}.csv file download successful")
        return csv_path
    
    def download_painting(self, base_dir=None, percent=100):
        print("Downloading data...")
        base_dir = base_dir or self.get_base_dir()

        objects_dimensions_csv = self.check_file_exists('objects_dimensions')
        objects_df = pd.read_csv(objects_dimensions_csv)
        
        published_images_csv = self.check_file_exists('published_images')
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
                # self.failed_object_ids.append(object_id)    

        print(f"Found {painted_df['objectid'].nunique()} images.")

        Parallel(n_jobs=16)(delayed(download_image)(object_id, thumb) for object_id, thumb in tqdm.tqdm(painted_df[['objectid', 'iiifthumburl']].values))
        # print(self.failed_object_ids)
        # painted_df = painted_df[~painted_df['objectid'].isin(self.failed_object_ids)]
        print(f"{len(painted_df['objectid'])} images download completed")
        
    def merge_and_filter(self, objects_df, images_df, output_file):
        painted_df = pd.merge(
            objects_df[['objectid', 'element']],
            images_df[['depictstmsobjectid', 'iiifthumburl']],
            left_on='objectid', right_on='depictstmsobjectid',
            how='inner'
        ).query("element == 'painted surface'")

        painted_df = painted_df.drop_duplicates().drop('depictstmsobjectid', axis=1)
        painted_df.to_csv(output_file + '/merged.csv', index=False)

        return painted_df


if __name__ == "__main__":
    curr_path = os.getcwd()
    loader_root = curr_path + "/data"
    percent = 100  
    downloader = ImageDownloader(loader_root)
    downloader.download_painting(percent=percent)
