"""
Writes images from nd2 files
"""

from pathlib import Path
import logging
import cv2
import nd2
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy.typing as npt
from nd2reader import ND2Reader

logger = logging.getLogger(__name__)


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
        "--bit-depths", choices=["8bit", "16bit"], default="16bit", type=str,
        help="What kind of image to write. Use '8bit' to view normally, use '16bit' for "
             "maximum accuracy. Default: %(default)s"
    )


def main(args):
    image(
        input_file=args.input_file,
        output_folder=args.output_folder,
        bit_depths=args.bit_depths
    )


def image(input_file, output_folder, bit_depths: str = "16bit"):
    nd2open = nd2.ND2File(input_file)

    # Extract iter axes names
    iter_axes = list(nd2open.sizes.keys())

    # Get images as arrays
    images = nd2open.asarray()

    # Make sure every image stack has all possible dimensions
    ALL_AXES = {"T", "C", "Z", "P", "X", "Y"}
    for missing_axes in ALL_AXES - set(iter_axes):
        iter_axes = [missing_axes] + iter_axes
        images = np.array([images])

    # sort array so input order is consistent (but make sure x/y is in the end)
    sort_idx = np.argsort(iter_axes[:-2])
    sort_idx = np.append(sort_idx, [4, 5])
    np.moveaxis(images, range(len(images.shape)), sort_idx)

    # Calculate min/max values for image conversion
    if bit_depths == "8bit":
        images_max = np.max(images)
        images_min = np.min(images)

    # make output folder
    output_folder.mkdir(parents=True, exist_ok=True)

    # Iterating over images
    len_c, len_p, len_t, len_z, len_x, len_y = images.shape
    for t in tqdm(range(len_t), desc="Extracting images"):
        for c in range(len_c):
            for z in range(len_z):
                for p in range(len_p):

                    # get image
                    img = images[c][p][t][z]

                    # Convert to 8bit
                    if bit_depths == "8bit":
                        img = convert_16bit_to_8bit(img, min_val=images_min,
                                                    max_val=images_max)

                    # Create filename
                    filename = output_folder.joinpath(
                        f"p-{p:02d}.z-{z:02d}.c-{c:02d}.t-{t:05d}.tif")

                    # Convert to PIL image and save in tif
                    img = Image.fromarray(img)
                    img.save(filename)

    nd2open.close()
    logger.info("Finished")


def convert_16bit_to_8bit(image16bit: npt.NDArray, min_val: int = None,
                          max_val: int = None) -> npt.NDArray:
    if not min_val:
        min_val = np.max(image16bit)

    if not max_val:
        max_val = np.min(image16bit)

    image8bit = ((image16bit - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    return image8bit
