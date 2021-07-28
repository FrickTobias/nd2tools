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


def main(args):
    with ND2Reader(args.input) as images:
        # TODO: Add filter for only z = n to movie
        # TODO: Add xy cropping capability

        output = generate_filename(raw_name=args.output)
        write_video_greyscale(file_path=output, frames=images, fps=args.fps,
                              cut=args.cut, trim=args.trimming, slice=args.slice)
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
def write_video_greyscale(file_path, frames, fps, cut=None, trim=None, slice=None):
    """
    Writes frames to an mp4 video file
    :param file_path: Path to output video, must end with .mp4
    :param frames: List of PIL.Image objects
    :param fps: Desired frame rate
    """

    # Adjusting output frame coordinates
    im_xy = ImageCoordinates(x1=0, x2=frames.sizes['x'], y1=0, y2=frames.sizes['y'])
    if slice:
        im_xy.slice(*slice)
        logger.info(
            f"Slicing frames to: x={im_xy.x1}:{im_xy.x2}, y={im_xy.y1}:{im_xy.y2}"
        )
    if cut:
        im_xy.cut_out(*cut)
        logger.info(
            f"Cutting frames to: x={im_xy.x1}:{im_xy.x2}, y={im_xy.y1}:{im_xy.y2}"
        )
    if trim:
        im_xy.trim(*trim)
        logger.info(
            f"Trimming frames to: x={im_xy.x1}:{im_xy.x2}, y={im_xy.y1}:{im_xy.y2}"
        )

    # Set up writer
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(filename=file_path, fourcc=fourcc, fps=fps,
                             frameSize=(im_xy.width(), im_xy.height()), isColor=0)

    for frame in tqdm(frames, desc=f"Saving images to movie: {file_path}",
                      unit=" frames"):
        if trim or cut or slice:
            frame = frame[im_xy.numpy_y1:im_xy.numpy_y2, im_xy.numpy_x1:im_xy.numpy_x2]

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
        "-t", "--trimming", nargs=4, type=int,
        metavar=("LEFT", "RIGHT", "BOTTOM", "TOP"),
        help="Trim images [pixels]. Done after --cut."
    )
    parser.add_argument(
        "-s", "--slice", type=int, nargs=2, metavar=("X_SPLITS", "Y_SPLITS"),
        help="Slice images."
    )


# TODO: Add capability of cropping by fraction?
class ImageCoordinates:
    """
                      frames.sizes['x']
         +-------------------'--------------------+
         '                                        '

         + - left - + --------------- + - right - +       -+
         |          |                 |           |        |
        top         |                 |          top       |
         |          |                 |           |        |
         + ------ x1,y2 ---- w ---- x2,y2 ------- +        |
         |          |                 |           |        |
         |          |                 |           |        |
         |          h                 h           |         } frames.sizes['y']
         |          |                 |           |        |
         |          |                 |           |        |
         + ------ x1,y1 ---- w ---- x2,y1 ------- +        |
         |          |                 |           |        |
       bottom       |                 |         bottom     |
         |          |                 |           |        |
         + - left - + --------------- + - right - +       -+


    """

    def __init__(self, x1, x2, y1, y2):
        # Never changing
        self.original_x1 = x1
        self.original_x2 = x2
        self.original_y1 = y1
        self.original_y2 = y2

        # Cartesian coordinates
        self.x1 = self.original_x1
        self.x2 = self.original_x2
        self.y1 = self.original_y1
        self.y2 = self.original_y2

        # Numpy conversions
        self.numpy_x1 = self.original_x1
        self.numpy_x2 = self.original_x2
        self.numpy_y1 = self.original_y2
        self.numpy_y2 = self.original_y1

    def cut_out(self, x1, x2, y1, y2):
        """
        Cuts out a part of frame
        """
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.update_numpy_coorinates()

    def trim(self, left, right, bottom, top):
        """
        Removes pixels from left, right, bottom and top of frame
        """
        self.x1 += left
        self.x2 -= right
        self.y1 += bottom
        self.y2 -= top
        self.update_numpy_coorinates()

    # TODO: Correct for slicing errors. Truncation?
    # TODO: Add possiblity to keep x/y
    def slice(self, x_splits, y_splits, x_keep=0, y_keep=0):
        """
        Splits frame into fractions and keeps one of them
        """
        x_chunk = int(self.width() / (x_splits + 1))
        y_chunk = int(self.height() / (y_splits + 1))
        self.x1 += x_keep * x_chunk
        self.x2 = self.x1 + x_chunk
        self.y1 += y_keep * y_chunk
        self.y2 = self.y1 + y_chunk
        self.update_numpy_coorinates()

    def update_numpy_coorinates(self):
        """
        Update numpy coordinates. Run after every time self.x1/x2/y1/y2 change.
        """
        self.numpy_y1 = self.original_y2 - self.y2
        self.numpy_y2 = self.original_y2 - self.y1
        self.numpy_x1 = self.x1
        self.numpy_x2 = self.x2

    def width(self):
        return self.x2 - self.x1

    def height(self):
        return self.y2 - self.y1
