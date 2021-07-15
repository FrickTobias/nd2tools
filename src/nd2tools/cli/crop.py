"""
Crops nd2 images
"""

import matplotlib.pyplot as plt
import pathlib
import logging
import cv2
from nd2reader import ND2Reader


logger = logging.getLogger(__name__)


def main(args):
    with ND2Reader(args.image) as images:
        print(images.sizes)
        images.iter_axes = args.iter_axes # TODO: Split this functionality into new submodule: nd2tools split <input.nd2> <out-prefix> => out-prefix.z-n.xy-n.<more>.png
        for iteration, frame in enumerate(images):

            # Scaling image, needed because image is in 16bit and png has max 256.
            frame_scaled = cv2.normalize(frame, dst=None, alpha=0, beta=65535,
                                       norm_type=cv2.NORM_MINMAX)

            outfile = f'{args.out_prefix}.{args.iter_axes}-{str(iteration)}.png'
            cv2.imwrite(outfile, frame_scaled)


# TODO: Make generator
# def nd2_generator(img_path):
#    with ND2Reader(img_path) as images:
#        yield images
#

def add_arguments(parser):
    parser.add_argument(
        "image", type=pathlib.Path,
        help="Input image"
    )
    parser.add_argument(
        "x1 y1", type=int, nargs=2,
        help="Two integers for first corner of cropping rectangle."
    )
    parser.add_argument(
        "x2 y2", type=int, nargs=2,
        help="Two integers for second corner of cropping rectangle."
    )
    parser.add_argument(
        "out_prefix",
        help="Output prefix. Final name will be <out_prefix>.<iter_axes>-n.png"
    )
    parser.add_argument(
        "--iter_axes", default='z',
        help="Axes to iterate over."
    )
