"""
Adds scalebars to images
"""

import matplotlib.pyplot as plt
import imageio
import pathlib
import logging
from argparse import ArgumentParser
from nd2reader import ND2Reader

from matplotlib_scalebar.scalebar import ScaleBar

logger = logging.getLogger(__name__)


def main(args):
    with ND2Reader(args.image) as images:
        print(images.sizes)
        images.iter_axes = 't'
        for frame in images:
            import pdb
            pdb.set_trace()
            plt.savefig(args.output, format="jpeg", bbox_inches='tight', pad_inches=0)


#def nd2_generator(img_path):
#    with ND2Reader(img_path) as images:
#        yield images


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
        "output",
        help="Output file name. Will save in jpeg."
    )
