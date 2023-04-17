"""
Writes images from nd2 files
"""

from pathlib import Path
import logging
import cv2
import nd2
import itertools
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy.typing as npt
from nd2reader import ND2Reader

logger = logging.getLogger(__name__)

NP_BIT_DTYPES = {
    "16bit": "uint16",
    "8bit": "uint8"
}


def add_arguments(parser):
    parser.add_argument(
        "--input-file", type=Path,
        help="Nd2 file"
    )
    parser.add_argument(
        "--output-folder", type=Path,
        help="Output folder name"
    )
    parser.add_argument(
        "--bit-depths", choices=["8bit", "16bit"], type=str,
        help="Convert image to another bit depths. Use '8bit' to have an image you can "
             "view normally but looses accuracy if input it '16bit'"
    )


def main(args):
    image(
        input_file=args.input_file,
        output_folder=args.output_folder,
        bit_depths=args.bit_depths
    )


def image(input_file, output_folder, bit_depths: str = None):
    # Open image file and create dask array (like np array but lazy loaded for memory optimization)
    with nd2.ND2File(input_file) as nd2open:
        images = nd2open.to_dask()

    # Compute max/min for conversions so all images have same conversion factor
    if bit_depths:
        bit_dtype = NP_BIT_DTYPES[bit_depths]
        if not images.dtype == bit_dtype:
            logger.info(f"Will convert image dtype from {images.dtype} to {bit_dtype}")
            images_max = images.max().compute()
            images_min = images.min().compute()

    # create output folder
    logger.info(f"Creating output: {output_folder}")
    output_folder.mkdir(parents=True, exist_ok=True)

    # Extract all information about channels in nd2 file
    channel_lengths = list(nd2open.sizes.values())[:-2]
    channels = list(nd2open.sizes.keys())[:-2]

    # Loops over all combination of all channels
    for channel_indices in tqdm(
            itertools.product(*[range(length) for length in channel_lengths]),
            desc="Extracting images", unit="img", total=np.product(channel_lengths)
    ):
        # Extract the combined channel and convert to np array
        image = np.array(images[channel_indices])

        # Construct the output filename
        if len(channel_indices) == 0:
            filename = "image.tif"
        else:
            filename = ".".join([f"{channels[idx]}-{val + 1}" for idx, val in
                                 enumerate(channel_indices)]) + ".tif"

        # Convert image to another format
        if bit_depths:
            if not image.dtype == bit_dtype:
                image = convert_bit_depths(image, bit_depths, images_min, images_max)

        # Convert to PIL Image and save
        Image.fromarray(image).save(output_folder.joinpath(filename))


def convert_bit_depths(image: npt.NDArray, bit_depths: str, min_val: int = None,
                       max_val: int = None) -> npt.NDArray:
    # Set conversion factor values
    if bit_depths == "8bit":
        bit_max = 255
        np_bit_depths = np.uint8
    elif bit_depths == "16bit":
        bit_max = 16383
        np_bit_depths = np.uint16

    # Calculate max/min value for image
    if not max_val:
        max_val = np.max(image)
    if not min_val:
        min_val = np.min(image)

    # Convert image
    image_converted = ((image - min_val) / (max_val - min_val) * bit_max).astype(
        np_bit_depths)

    return image_converted
