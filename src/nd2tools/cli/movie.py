"""
Converts images to MP4 movies
"""

import pathlib
import logging
import cv2
import sys
import pims
import numpy as np
from tqdm import tqdm
from nd2reader import ND2Reader

logger = logging.getLogger(__name__)
EXCLUSION_LIST = list('x' 'y')


def main(args):
    with ND2Reader(args.input) as images:
        # TODO: Add filter for only z = n to movie
        # TODO: Add xy cropping capability

        output = generate_filename(raw_name=args.output)
        write_video_greyscale(file_path=output, frames=images, fps=args.fps)
        logger.info(f"Finished")


def generate_filename(raw_name, metadata=False):
    name, format = adjust_for_file_extension(raw_name)

    if metadata:
        output = f"{name}.{metadata}.{format}"
    else:
        output = f"{name}.{format}"

    return output


def adjust_for_file_extension(filename, default_format="mp4",
                              accepted_extensions=("mp4", "MP4")):
    # No extension
    if "." not in filename:
        return filename, default_format

    # Check extension is accepted (otherwise adds default)
    name, format = filename.rsplit(".", 1)
    if format not in accepted_extensions:
        return filename, default_format
    else:
        return name, format


#
# Adapted from cv2.VideoWriter() examples.
# https://www.programcreek.com/python/example/72134/cv2.VideoWriter
def write_video_greyscale(file_path, frames, fps):
    """
    Writes frames to an mp4 video file
    :param file_path: Path to output video, must end with .mp4
    :param frames: List of PIL.Image objects
    :param fps: Desired frame rate
    """

    w, h = frames.sizes['x'], frames.sizes['y']
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(file_path, fourcc, fps, (w, h), 0)

    for frame in tqdm(frames, desc=f"Saving images to movie: {file_path}",
                      unit=" frames"):
        frame_8bit = map_uint16_to_uint8(frame)
        writer.write(frame_8bit)

    writer.release()


#
# Modified from stackoverflow
# https://stackoverflow.com/questions/25485886/how-to-convert-a-16-bit-to-an-8-bit-image-in-opencv
def map_uint16_to_uint8(img, lower_bound=None, upper_bound=None):
    '''
    Map a 16-bit image trough a lookup table to convert it to 8-bit.

    Parameters
    ----------
    img: numpy.ndarray[np.uint16]
        image that should be mapped
    lower_bound: int, optional
        lower bound of the range that should be mapped to ``[0, 255]``,
        value must be in the range ``[0, 65535]`` and smaller than `upper_bound`
        (defaults to ``numpy.min(img)``)
    upper_bound: int, optional
       upper bound of the range that should be mapped to ``[0, 255]``,
       value must be in the range ``[0, 65535]`` and larger than `lower_bound`
       (defaults to ``numpy.max(img)``)

    Returns
    -------
    numpy.ndarray[uint8]
    '''
    if lower_bound is None:
        lower_bound = np.min(img)
    if upper_bound is None:
        upper_bound = np.max(img)

    lut = np.concatenate([
        np.zeros(lower_bound, dtype=np.uint16),
        np.linspace(0, 255, upper_bound - lower_bound).astype(np.uint16),
        np.ones(2 ** 16 - upper_bound, dtype=np.uint16) * 255
    ])
    return lut[img].astype(np.uint8)


#
# Copied from stackoverflow issue
# https://stackoverflow.com/questions/21596281/how-does-one-convert-a-grayscale-image-to-rgb-in-opencv-python
def gray_to_rgb(image):
    gray_three = cv2.merge([image, image, image])
    return gray_three


def add_arguments(parser):
    parser.add_argument(
        "input", type=pathlib.Path,
        help="Input PNG image"
    )
    parser.add_argument(
        "-o", "--output",
        help="Write to MP4 file."
    )
    parser.add_argument(
        "-f", "--fps", type=int, default=10,
        help="Frames per second in output movie."
    )
