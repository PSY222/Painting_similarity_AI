import os
import pandas as pd
import urllib.request
import tqdm
from joblib import Parallel, delayed
from math import floor

#inspired by https://github.com/LukeWood/goa-loader/blob/master/goa_loader/download_data.py

class ImageDownloader:
    def __init__(self, loader_root, csv_remote_path):
        self.loader_root = loader_root
        self.csv_remote_path = csv_remote_path

    def ensure_exists(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def get_base_dir(self):
        self.ensure_exists(self.loader_root)
        self.ensure_exists(f"{self.loader_root}/annotations")
        self.ensure_exists(f"{self.loader_root}/images")
        return self.loader_root

    def thumbnail_to_local(self, base_path, thumb):
        image_path = f"{base_path}/images"
        ending = "_".join(thumb.split("/")[-5:])
        return f"{image_path}/{ending}"

    def get_file(self, remote_url, out, timeout_seconds=10):
        with urllib.request.urlopen(remote_url, timeout=timeout_seconds) as response:
            with open(out, "wb") as out_file:
                data = response.read()  # a `bytes` object
                out_file.write(data)

    def download(self, base_dir=None, percent=100, num_images=None):
        print("Downloading data...")
        base_dir = base_dir or self.get_base_dir()
        csv_file = f"{base_dir}/annotations/published_images.csv"

        if not os.path.exists(csv_file):
            print(f"CSV not found, downloading from {self.csv_remote_path}")
            self.get_file(self.csv_remote_path, out=csv_file, timeout_seconds=100)
            print(".csv file download successful")

        print(f"Reading annotations from {csv_file}")
        df = pd.read_csv(csv_file)

        samples = floor(df.shape[0] * (percent / 100))
        if num_images:
            samples = min(samples, num_images)
        print(f"Found {df.iiifthumburl.nunique()} images.")
        print(f"Downloading {samples}/{df.shape[0]} images")

        df = df.head(samples)
        print("Downloading images...")

        def download_image(thumb):
            out = self.thumbnail_to_local(base_dir, thumb)
            if os.path.exists(out):
                return

            try:
                self.get_file(thumb, out=out)
            except Exception as e:
                print(e)
                print(f"failed to get {thumb}")

        Parallel(n_jobs=16)(
            delayed(download_image)(thumb) for thumb in tqdm.tqdm(df.iiifthumburl.unique())
        )
        print("Image download completed")

if __name__ == "__main__":
    curr_path = os.getcwd()
    loader_root = curr_path + "/data"
    csv_remote_path = "https://raw.githubusercontent.com/NationalGalleryOfArt/opendata/main/data/published_images.csv"
    percent = 100  
    num_images = 200  # Number of images to download

    downloader = ImageDownloader(loader_root, csv_remote_path)
    downloader.download(percent=percent, num_images=num_images)
