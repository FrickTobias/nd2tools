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

from nd2tools.utils import ImageCoordinates

logger = logging.getLogger(__name__)


def main(args):
    with ND2Reader(args.input) as images:
        output = generate_filename(raw_name=args.output)

        # Adjusting output frame coordinates
        im_xy = ImageCoordinates(x1=0, x2=images.sizes['x'], y1=0, y2=images.sizes['y'])
        im_xy = adjust_frame(im_xy, args.split, args.cut, args.trim)
        write_video_greyscale(file_path=output, frames=images, fps=args.fps,
                              width=im_xy.width(), height=im_xy.height(),
                              crop_x1=im_xy.np_x1, crop_x2=im_xy.np_x2,
                              crop_y1=im_xy.np_y1, crop_y2=im_xy.np_y2)
        logger.info(f"Finished")


def adjust_frame(image_coordinates, split, cut, trim):
    """
    Adjusts xy coordinates of image.
    """
    if split:
        image_coordinates.split(*split)
        logger.info(
            f"Slicing frames to: x={image_coordinates.x1}:{image_coordinates.x2}, "
            f"y={image_coordinates.y1}:{image_coordinates.y2}"
        )
    if cut:
        image_coordinates.cut_out(*cut)
        logger.info(
            f"Cutting frames to: x={image_coordinates.x1}:{image_coordinates.x2}, "
            f"y={image_coordinates.y1}:{image_coordinates.y2}"
        )
    if trim:
        image_coordinates.trim(*trim)
        logger.info(
            f"Trimming frames to: x={image_coordinates.x1}:{image_coordinates.x2}, "
            f"y={image_coordinates.y1}:{image_coordinates.y2}"
        )

    return image_coordinates


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


def write_video_greyscale(file_path, frames, fps, width, height, crop_x1=None,
                          crop_x2=None, crop_y1=None, crop_y2=None):
    """
    Writes frames to an mp4 video file
    :param file_path: Path to output video, must end with .mp4
    :param frames: List of PIL.Image objects
    :param fps: Desired frame rate
    :param cropping_tuple: numpy format xy coordinates for cropping
    """

    # Writing movie
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(filename=file_path, fourcc=fourcc, fps=fps,
                             frameSize=(width, height), isColor=0)
    for frame in tqdm(frames, desc=f"Saving images to movie: {file_path}",
                      unit=" frames"):
        frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        # TODO: Fix color scaling between frames
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
        "-f", "--fps", type=int, default=4,
        help="Frames per second in output movie."
    )
    parser.add_argument(
        "-c", "--cut", nargs=4, type=int,
        metavar=("x1", "x2", "y1", "y2"),
        help="Cut out frame x1y1:x2y2."
    )
    parser.add_argument(
        "-t", "--trim", nargs=4, type=int,
        metavar=("LEFT", "RIGHT", "BOTTOM", "TOP"),
        help="Trim images [pixels]. Done after --cut."
    )
    parser.add_argument(
        "-s", "--split", type=int, nargs=2, metavar=("X_PIECES", "Y_PIECES"),
        help="Splits images."
    )
