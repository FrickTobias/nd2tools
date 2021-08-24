"""
Writes png images from nd2 files
"""

import matplotlib.pyplot as plt
import imageio
import pathlib
import logging
import numpy as np
import cv2

from matplotlib_scalebar.scalebar import ScaleBar
from nd2reader import ND2Reader

from nd2tools.utils import ImageCoordinates
from nd2tools.utils import map_uint16_to_uint8
from nd2tools.utils import generate_filename
from nd2tools.utils import ScalingMinMax
from nd2tools.utils import add_global_args
from nd2tools.utils import get_screen_dpi

logger = logging.getLogger(__name__)


def add_arguments(parser):
    add_global_args(parser)
    parser.add_argument(
        "input", type=pathlib.Path,
        help="Nd2 file"
    )
    parser.add_argument(
        "output",
        help="Output file name. Will save in PNG."
    )


def main(args):
    image(input=args.input, output=args.output, split=args.split, keep=args.keep,
          cut=args.cut, trim=args.trim)


def image(input, output, split, keep, cut, trim):
    with ND2Reader(input) as images:
        im_xy = ImageCoordinates(x1=0, x2=images.sizes['x'], y1=0, y2=images.sizes['y'])
        im_xy.adjust_frame(split, keep, cut, trim)
        frame_pos_list = im_xy.frames()
        scaling_min_max = ScalingMinMax(mode="continuous",
                                        scaling=1,
                                        image=images[0])

        for frame_number, image in enumerate(images):

            # ims = list()
            for frame_fraction, frame_pos in enumerate(frame_pos_list):

                # Crop image
                x1, x2, y1, y2 = frame_pos
                image_crop = image[y1:y2, x1:x2]

                # convert 16bit to 8bit
                if image_crop.dtype == "uint16":
                    if scaling_min_max.mode == "continuous" or scaling_min_max.mode == "current":
                        logger.info(f"frame: {frame_number}")
                        scaling_min_max.update(image_crop)
                    image_crop = map_uint16_to_uint8(image_crop,
                                                     lower_bound=scaling_min_max.min_current,
                                                     upper_bound=scaling_min_max.max_current)

                metadata = list()
                if len(images) >= 2:
                    metadata.append(f"image-{frame_number + 1}")
                if len(frame_pos_list) >= 2:
                    metadata.append(f"frame-{frame_fraction + 1}")
                if len(metadata) >= 1:
                    metadata = ".".join(metadata)
                else:
                    metadata = False

                file_path = generate_filename(output, metadata=metadata,
                                              format="png")

                cv2.imwrite(file_path, image_crop)