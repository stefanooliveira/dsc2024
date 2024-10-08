import os
import json
from pathlib import Path
from typing import Optional, List, Tuple
from functools import lru_cache

import pandas

from dsc2024 import handling

_base_path = os.path.dirname(os.path.dirname(__file__))
_datasets_dir = os.environ.get(
    "DSC2024_DATASETS_DIR",
    os.path.join(_base_path, "datasets")
)
datasets_dir = Path(_datasets_dir)
dataset_default = "public.csv"
vector_images_filepath = "images_cropped_full_samples_2000km.pkl.xz"

# ref: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
pandas.options.mode.copy_on_write = True




@lru_cache
def get_public_dataset(
    parse_hora_ref: bool = True,
    expand_metar_and_metaf: bool = True,
    add_image_vectors: bool = True,
    add_anac_extra_info: bool = True,
    sampling: Optional[int] = None,
    set_flightid_as_index: bool = True
) -> pandas.DataFrame:
    df = pandas.read_csv(datasets_dir / dataset_default)
    #df = df.sample(n=30000, random_state=42)
    print(f"Dataset {dataset_default} has shape {df.shape} with images imported from {vector_images_filepath} and converted to vectors")
    
    if parse_hora_ref:
        df.hora_ref = handling.parse_hora_ref_as_series(df.hora_ref)
    if add_anac_extra_info:
        df_anac = get_anac_aerodromos_publicos()
        df = handling.add_anac_extra_info(df, df_anac)
    if expand_metar_and_metaf:
        df_metar_extra = get_metar_extra()
        df = handling.expand_metar_and_metaf_features(df, df_metar_extra)
    if set_flightid_as_index:
        df.set_index("flightid", inplace=True)
    if add_image_vectors:
        df_image = get_image_embedding()
        df = handling.add_image_vectors(df, df_image)
    return df


def _generate_raw_data_kwargs(raw_data: bool = True):
    """Generate arguments to not add extra parsing"""
    kwargs = {}
    if raw_data:
        kwargs = {
            "expand_metar_and_metaf": False,
            "set_flightid_as_index": False,
            "parse_hora_ref": False,
            "add_image_vectors": False,
            "add_anac_extra_info": False
        }
    return kwargs


def get_train_dataset(sampling: Optional[int] = None,
                      raw_data: bool = False):
    """Get all rows with 'espera' labeled as 0 or 1"""
    raw_kwargs = _generate_raw_data_kwargs(raw_data)
    df = get_public_dataset(sampling=sampling, **raw_kwargs)
    mask = df.notna().espera
    return df[mask]


def get_test_dataset(sampling: Optional[int] = None,
                     raw_data: bool = False):
    """Only get rows with 'espera' with missing-values"""
    raw_kwargs = _generate_raw_data_kwargs(raw_data)
    df = get_public_dataset(sampling=sampling, **raw_kwargs)
    mask = df.isna().espera
    return df[mask]


def save_image_embedding(df: pandas.DataFrame):
    image_embedding_path = datasets_dir / "images_cropped_full_samples_2000km.pkl.xz"
    df.to_pickle(image_embedding_path)
    print(f"Saved pickle at {image_embedding_path}")


def get_image_embedding() -> pandas.DataFrame:
    image_embedding_path = datasets_dir / vector_images_filepath
    print(f"Reading from {image_embedding_path}")
    return pandas.read_pickle(image_embedding_path)


@lru_cache
def get_image_mask_points() -> List[Tuple[int, int]]:
    fpath = datasets_dir / "image-mask" / "image-mask.json"
    with fpath.open(encoding="UTF-8") as source:
        image_mask = json.load(source)[0]
    return [
        (round(p["x"]), round(p["y"]))
        for p in image_mask["content"]
    ]


@lru_cache
def get_metar_extra() -> pandas.DataFrame:
    fpath = datasets_dir / "metar" / "metar_extra.csv"
    df = pandas.read_csv(fpath, parse_dates=["valid"])
    # truncate hour
    df.valid = df.valid.dt.floor("h")
    df.set_index(["station", "valid"], inplace=True)
    df.sort_index(inplace=True)
    return df


def get_anac_aerodromos_publicos() -> pandas.DataFrame:
    fpath = datasets_dir / "anac" / "aerodromos_publicos.csv"
    columns_map = {
        "OACI": "id",
        "Altitude": "altitude",
        "Operação Diurna": "op_diurna",
        "Operação Noturna": "op_noturna",
        "Designação 1": "designacao",
        "Comprimento 1": "comprimento",
        "Largura 1": "largura",
        "Superfície 1": "superficie"
    }
    df = pandas.read_csv(fpath)
    df.rename(columns=columns_map, inplace=True)
    df = df[list(columns_map.values())]
    df[["designacao_left", "designacao_right"]] = df["designacao"].str.split("/", n=1, expand=True)
    df.drop("designacao", axis=1, inplace=True)
    df.set_index("id", inplace=True)
    return df


if __name__ == '__main__':
    get_public_dataset()