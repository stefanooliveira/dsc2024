from typing import Optional
from dataclasses import dataclass
import os


from tqdm_joblib import tqdm_joblib
from joblib import Memory
from loguru import logger
import joblib
import numpy as np
import pandas as pd
import requests


from dsc2024 import datasets
from dsc2024 import features
from dsc2024 import images



DEFAULT_CROP_RANGE = 2000


# FIXME(@lerax): ter 23 abr 2024 14:43:39
# global variables are only acceptable here because this is a script
# but would be nice to refactor this
logger.debug("loading vision transformers as global variables")
MASK = images.generate_pil_crop_mask(datasets.get_image_mask_points())
PREPROCESSOR, VIT = features.load_transformer_feature_extractor()
logger.debug("transformers loaded!")
MEMORY = Memory(datasets.datasets_dir / 'cache', verbose=0)


@dataclass
class FlightImageVector:
    flightid: str
    vector: Optional[np.ndarray]


def download_flight_image_vector(flightid: str, url: Optional[str], flight_destination) -> FlightImageVector:
    if not isinstance(url, str) or not url:
        return FlightImageVector(flightid=flightid, vector=None)
    img = images.download_image_and_cropit(url, flight_destination, DEFAULT_CROP_RANGE)
    if img is None:
        return FlightImageVector(flightid=flightid, vector=None)

# @HACK
    # import matplotlib.pyplot as plt
    # import requests
    # from PIL import Image
    # from io import BytesIO

    # # Download the image using requests
    # response = requests.get(url)
    # original_image = Image.open(BytesIO(response.content))

    # plt.figure(figsize=(15, 10))

    # # First plot
    # plt.subplot(1, 2, 2)
    # plt.imshow(original_image)
    # plt.title('Imagem Original')
    # plt.xlabel('Longitude')
    # plt.ylabel('Latitude')

    # # Second plot
    # plt.subplot(1, 2, 1)
    # plt.imshow(img)
    # latitude = round(images.cropper.get_airport_location_by_icao(flight_destination)[0],4)
    # longitude = round(images.cropper.get_airport_location_by_icao(flight_destination)[1],4)
    # plt.title(f'Coordenadas para {flight_destination}: ({latitude}, {longitude}) com {DEFAULT_CROP_RANGE}km^2')
    # plt.xlabel('Longitude')
    # plt.ylabel('Latitude')

    # plt.tight_layout()
    # plt.show()
# @HACK


    vector = features.feature_extraction_from_image(img, PREPROCESSOR, VIT)
    return FlightImageVector(
        flightid=flightid,
        vector=vector
    )


def create_delayed_tasks(df: pd.DataFrame):
    return [
        joblib.delayed(download_flight_image_vector)(t.flightid, t.url_img_satelite, t.destino)
        for t in df.itertuples()
    ]


def download_batch_parallel():
    raw_kwargs = datasets._generate_raw_data_kwargs()
    logger.info("loading dataset with urls for download")
    df = datasets.get_public_dataset(**raw_kwargs)
    n_jobs = os.cpu_count() / 2  # in parallel, use half of CPU cores
    tasks = create_delayed_tasks(df)
    n_tasks = len(tasks)
    logger.info(f"total of jobs: {len(tasks)} | in parallel: {n_jobs}")
    logger.info("starting to download the image vectors")
    with tqdm_joblib(desc="image-vectors", total=n_tasks):
        parallel_pool = joblib.Parallel(n_jobs=n_jobs, prefer="threads")
        df_vectors = pd.DataFrame(parallel_pool(tasks))

    logger.info("saving dataframe")
    datasets.save_image_embedding(df_vectors)


if __name__ == "__main__":
    download_batch_parallel()
    #@HACK
    # url_test = 'http://satelite.cptec.inpe.br/repositoriogoes/goes16/goes16_web/ams_ret_ch11_baixa/2022/06/S11635384_202206010100.jpg'
    # flightid = '504a62621cd231d6ab67e674ce538cd3'
    # destination = 'SBFL'
    # download_flight_image_vector(flightid, url_test, destination)
    #@HACK
