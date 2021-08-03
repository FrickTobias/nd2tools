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
from nd2tools.utils import generate_filename

logger = logging.getLogger(__name__)


# TODO: Update documentation


def add_arguments(parser):
    parser.add_argument(
        "input", type=pathlib.Path,
        help="Input PNG image"
    )
    parser.add_argument(
        "output",
        help="Output MP4 file."
    )
    parser.add_argument(
        "-f", "--fps", type=int, default=30,
        help="Frames per second. Default: %(default)s"
    )
    parser.add_argument(
        "-c", "--cut", nargs=4, type=int,
        metavar=("X1", "X2", "Y1", "Y2"),
        help="Cut out frame x1y1:x2y2."
    )
    parser.add_argument(
        "-t", "--trim", nargs=4, type=int,
        metavar=("LEFT", "RIGHT", "BOTTOM", "TOP"),
        help="Trim images [pixels]."
    )
    parser.add_argument(
        "-s", "--split", type=int, nargs=2, metavar=("X_PIECES", "Y_PIECES"),
        help="Splits images into a X_PIECES x Y_PIECES grid. See -k (--keep) for which "
             "piece is saved."
    )
    parser.add_argument(
        "-k", "--keep", nargs=2, default=["1", "1"], metavar=("X_PIECE", "Y_PIECE"),
        help="Specify which slice to keep. Use 0 to keep all and save to "
             "OUTPUT.frame-N.mp4. %(default)s."
    )
    parser.add_argument(
        "--scaling", choices=["fast", "continuous", "independant", "naive"],
        default="fast",
        help="This option determined how min/max are set for conversion from 16bit to "
             "8bit color space. fast: first_image. continuous: "
             "images_read(frame_number). independant: current_image. naive: Uses min=0 "
             "and max=2^16-1. Default: %(default)s."
    )


def main(args):
    with ND2Reader(args.input) as images:
        # Adjusting output frame coordinates
        im_xy = ImageCoordinates(x1=0, x2=images.sizes['x'], y1=0, y2=images.sizes['y'])
        im_xy = adjust_frame(im_xy, args.keep, args.split, args.cut, args.trim)

        frame_pos_list = im_xy.frames()
        write_video_greyscale(file_prefix=args.output, images=images, fps=args.fps,
                              width=im_xy.frame_width(), height=im_xy.frame_height(),
                              frame_pos_list=frame_pos_list,
                              scaling_method=args.scaling)
        logger.info("Finished")


def adjust_frame(image_coordinates, keep, split, cut, trim):
    """
    Adjusts xy coordinates of image.
    """
    if split:
        keep_0_index = [int(xy) - 1 for xy in keep]
        x, y = keep_0_index
        image_coordinates.split(*split, x_keep=x, y_keep=y)
        logger.info(
            f"Slicing image into a {split} grid"
        )
    if cut:
        image_coordinates.cut_out(*cut)
        logger.info(
            f"Cutting frames to: x={image_coordinates.x1()}:{image_coordinates.x2()}, "
            f"y={image_coordinates.y1()}:{image_coordinates.y2()}"
        )
    if trim:
        image_coordinates.trim(*trim)
        logger.info(
            f"Trimming frames to: x={image_coordinates.x1()}:{image_coordinates.x2()}, "
            f"y={image_coordinates.y1()}:{image_coordinates.y2()}"
        )

    return image_coordinates


def write_video_greyscale(file_prefix, images, fps, width, height,
                          frame_pos_list, scaling_method="fast"):
    """
    Writes images to an mp4 video file
    :param file_path: Path to output video, must end with .mp4
    :param images: List of PIL.Image objects
    :param fps: Desired frame rate
    :param cropping_tuple: numpy format xy coordinates for cropping
    """

    # Opens one file per frame_pos tracks using open_video_files.dict[frame_pos] = writer
    open_video_files = OpenVideoFiles(file_prefix, fps, width, height, frame_pos_list)

    # Writing outputs
    scaling_min_max = ScalingMinMax(mode=scaling_method, images=images)
    for image in tqdm(images, desc=f"Writing movie file(s)", unit=" images"):
        if scaling_min_max.mode == "continuous":
            scaling_min_max.update(image)

        # Split image and writes to appropriate files
        for frame_pos in frame_pos_list:
            x1, x2, y1, y2 = frame_pos
            image_cropped = image[y1:y2, x1:x2]
            image_cropped_8bit = map_uint16_to_uint8(image_cropped,
                                                     lower_bound=scaling_min_max.min,
                                                     upper_bound=scaling_min_max.max)
            open_video_files.dictionary[frame_pos].write(image_cropped_8bit)

    # Close files
    open_video_files.close()


class OpenVideoFiles:

    def __init__(self, file_prefix, fps, width, height, frame_pos_list, is_color=0):

        # Opens one file per frame position set
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.dictionary = dict()
        for count, frame_pos in enumerate(frame_pos_list):

            # If multiple output, add frame-N to file name
            if len(frame_pos_list) >= 2:
                metadata = f"frame-{count + 1}"
            else:
                metadata = False
            file_path = generate_filename(file_prefix, metadata=metadata)

            # Open writers and track open files using dictionary
            logger.info(f"Creating file: {file_path}")
            writer = cv2.VideoWriter(filename=file_path, fourcc=fourcc, fps=fps,
                                     frameSize=(width, height), isColor=is_color)
            self.dictionary[frame_pos] = writer

    def close(self):
        for writer in self.dictionary.values():
            writer.release()


class ScalingMinMax:

    def __init__(self, mode, images):

        self.mode = mode
        if self.mode == "fast" or self.mode == "continuous":
            self.min = np.min(images[0])
            self.max = np.max(images[0])
        elif self.mode == "independant":
            self.min = None
            self.max = None
        elif self.mode == "naive":
            self.min = 0
            self.min = 65535  # = 16^2 - 1

    def update(self, image):
        frame_min = np.min(image[0])
        frame_max = np.max(image[0])
        self.min = min(self.min, frame_min)
        self.max = max(self.max, frame_max)
