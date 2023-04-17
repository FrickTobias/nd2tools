"""
Writes mp4 videos from nd2 files
"""

import cv2
import logging
import itertools
import matplotlib
import numpy as np
from tqdm import tqdm
from pathlib import Path
import numpy.typing as npt

import nd2

from nd2tools import utils

logger = logging.getLogger(__name__)


def add_arguments(parser):
    parser.add_argument(
        "--input-file", type=Path, required=True,
        help="Name of input ND2 file."
    )
    parser.add_argument(
        "--output-folder", type=Path, required=True,
        help="Name of output folder for mp4 files."
    )
    parser.add_argument(
        "--fps", type=float, default=30,
        help="Frames per second. Default: %(default)s"
    )


def main(args):
    movie(input_file=args.input_file, output_folder=args.output_folder,
          fps=args.fps)

def movie(input_file: Path, output_folder: Path, fps: int = 30):
    # Open image file and create dask array (like np array but lazy loaded for memory
    # optimization)
    with nd2.ND2File(input_file) as nd2open:
        images = nd2open.to_dask()

    # Compute max/min for conversions so all images have same conversion factor
    if images.dtype == "uint16":
        images_max = images.max().compute()
        images_min = images.min().compute()

    # Get frame dimensions
    channels_info = nd2open.sizes
    w = channels_info.pop("X")
    h = channels_info.pop("Y")

    # Get channel information
    channel_lengths = list(channels_info.values())
    channels = list(channels_info.keys())

    # Get channel information divided by time/non-time for filenames
    if "T" in channels_info:
        time_idx = np.where(np.array(list(channels_info.keys())) == "T")[0][0]
        channels_info_no_time = channels_info.copy()
        channels_info_no_time.pop("T")
        channel_no_time_lengths = list(channels_info_no_time.values())
        channels_no_time = list(channels_info_no_time.keys())
    else:
        time_idx = [-1]
        channel_no_time_lengths = channel_lengths

    # Create output dir
    logger.info(f"Creating outdir: {output_folder}")
    output_folder.mkdir(parents=True, exist_ok=True)

    # Open mp4 files; one per combination of channels (except time, x and y axis)
    mp4_writers = {}
    FOURCC = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    for channel_indices_no_time in itertools.product(
            *[range(length) for length in channel_no_time_lengths]):

        # Create filename string
        if len(channel_indices_no_time) == 0:
            filename = "movie.mp4"
        else:
            filename = ".".join([f"{channels_no_time[idx]}-{val + 1}" for idx, val in
                                 enumerate(channel_indices_no_time)]) + ".mp4"
        filename = str(output_folder.joinpath(filename))

        # Setup mp4 writers
        logger.info(f"Creating file: {filename}")
        writer = cv2.VideoWriter(filename=filename, fourcc=FOURCC, fps=fps,
                                 frameSize=(w, h), isColor=False)

        # Save in dictionary
        mp4_writers[channel_indices_no_time] = writer

    # Loops over all combination of all channels
    for channel_indices in tqdm(
            itertools.product(*[range(length) for length in channel_lengths]),
            desc="Extracting images", unit="img", total=np.product(channel_lengths)
    ):
        # Extract image for current channel combination and convert to np array
        image = np.array(images[channel_indices])

        # Fetch writer ID
        if time_idx != [-1]:
            writer_index = list(channel_indices)
            writer_index.pop(time_idx)
            writer_index = tuple(writer_index)
        else:
            writer_index = channel_indices

        # Make sure image is 8bit
        if image.dtype == "uint16":
            image = utils.convert_bit_depths(image, "8bit", images_min, images_max)

        # Write image as frame to mp4
        mp4_writers[writer_index].write(image)

    # Close mp4 files
    logger.info("Closing files")
    for writer in mp4_writers.values():
        writer.release()

    logger.info("Finished")
