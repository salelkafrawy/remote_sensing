from pathlib import Path

import numpy as np
from PIL import Image
import tifffile
import torch


def load_patch(
    observation_id,
    patches_path,
    use_ffcv_loader,
    *,
    data="all",
    landcover_mapping=None,
    return_arrays=True
):
    """Loads the patch data associated to an observation id

    Parameters
    ----------
    observation_id : integer
        Identifier of the observation.
    patches_path : string / pathlib.Path
        Path to the folder containing all the patches.
    data : string or list of string
        Specifies what data to load, possible values: 'all', 'rgb', 'near_ir', 'landcover' or 'altitude'.
    landcover_mapping : 1d array-like
        Facultative mapping of landcover codes, useful to align France and US codes.
    return_arrays : boolean
        If True, returns all the patches as Numpy arrays (no PIL.Image returned).

    Returns
    -------
    patches : tuple of size 4 containing 2d array-like objects
        Returns a tuple containing all the patches in the following order: RGB, Near-IR, altitude and landcover.
    """
    observation_id = str(observation_id)

    region_id = observation_id[0]
    if region_id == "1":
        region = "patches-fr"
    elif region_id == "2":
        region = "patches-us"
    else:
        raise ValueError(
            "Incorrect 'observation_id' {}, can not extract region id from it".format(
                observation_id
            )
        )

    subfolder1 = observation_id[-2:]
    subfolder2 = observation_id[-4:-2]

    filename = Path(patches_path) / region / subfolder1 / subfolder2 / observation_id

    patches = {}

    if data == "all":
        data = ["rgb", "near_ir", "altitude", "landcover"]

    if "rgb" in data:
        rgb_filename = filename.with_name(filename.stem + "_rgb.jpg")
        rgb_patch = Image.open(rgb_filename)

    return rgb_patch
