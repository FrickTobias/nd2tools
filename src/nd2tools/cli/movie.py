"""
Converts images to MP4 movies
"""

import pathlib
import logging
import cv2
import numpy as np
from tqdm import tqdm
from nd2reader import ND2Reader

from nd2tools.utils import ImageCoordinates
from nd2tools.utils import map_uint16_to_uint8

logger = logging.getLogger(__name__)


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
    parser.add_argument(
        "--scaling", choices=["fast", "continuous", "independant", "naive"],
        default="fast",
        help="This option determined how min/max are set for conversion from 16bit to "
             "8bit color space. fast: first_image. continuous: "
             "images_read(frame_number). independant: current_image. naive: Uses min=0 "
             "and max=2^16-1. Default: %(default)s."
    )
    # TODO: Add option for keeping specific xy/slicing window (or all)


def main(args):
    with ND2Reader(args.input) as images:
        output = generate_filename(raw_name=args.output)

        # Adjusting output frame coordinates
        im_xy = ImageCoordinates(x1=0, x2=images.sizes['x'], y1=0, y2=images.sizes['y'])
        im_xy = adjust_frame(im_xy, args.split, args.cut, args.trim)
        write_video_greyscale(file_path=output, frames=images, fps=args.fps,
                              width=im_xy.width(), height=im_xy.height(),
                              scaling=args.scaling, crop_x1=im_xy.np_x1,
                              crop_x2=im_xy.np_x2, crop_y1=im_xy.np_y1,
                              crop_y2=im_xy.np_y2)
        logger.info("Finished")


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


def write_video_greyscale(file_path, frames, fps, width, height, scaling, crop_x1=None,
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
    scaling_min_max = ScalingMinMax(mode=scaling, frames=frames)
    for frame in tqdm(frames, desc=f"Saving movie to file: {file_path}",
                      unit=" frames"):
        frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        if scaling_min_max.mode == "continuous":
            scaling_min_max.update(frame)

        frame_8bit = map_uint16_to_uint8(frame, lower_bound=scaling_min_max.min,
                                         upper_bound=scaling_min_max.max)
        writer.write(frame_8bit)

    writer.release()


class ScalingMinMax:

    def __init__(self, mode, frames):

        self.mode = mode
        if self.mode == "fast" or self.mode == "continuous":
            self.min = np.min(frames[0])
            self.max = np.max(frames[0])
        elif self.mode == "independant":
            self.min = None
            self.max = None
        elif self.mode == "naive":
            self.min = 0
            self.min = 65535  # = 16^2 - 1

    def update(self, frame):
        frame_min = np.min(frame[0])
        frame_max = np.max(frame[0])
        self.min = min(self.min, frame_min)
        self.max = max(self.max, frame_max)
