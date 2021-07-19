"""
Crops nd2 images
"""

import matplotlib.pyplot as plt
import pathlib
import logging
import cv2
import pims
from nd2reader import ND2Reader

logger = logging.getLogger(__name__)


def main(args):
    image = cv2.imread(args.input)

    logger.info("Cropping image")
    image_crop = pims.process.crop(image, (
    (args.trim_left, args.trim_right), (args.trim_top, args.trim_bottom), (0, 0)))

    logger.info(f'Writing to file {args.output}')
    cv2.imwrite(args.output, image_crop)


def add_arguments(parser):
    parser.add_argument(
        "input", type=str,
        help="Input .nd2 image"
    )
    parser.add_argument(
        "trim_left", type=int,
        help="Number of pixels to trim from left."
    )
    parser.add_argument(
        "trim_right", type=int,
        help="Number of pixels to trim from right."
    )
    parser.add_argument(
        "trim_top", type=int,
        help="Number of pixels to trim from top."
    )
    parser.add_argument(
        "trim_bottom", type=int,
        help="Number of pixels to trim from bottom."
    )
    parser.add_argument(
        "-o", "--output",
        help="Write to PNG file."
    )
