"""
Splits nd2 images into individual images.
"""

import matplotlib.pyplot as plt
import pathlib
import logging
import cv2
from nd2reader import ND2Reader

logger = logging.getLogger(__name__)
EXCLUSION_LIST = list('x' 'y')


def main(args):
    with ND2Reader(args.nd2_image) as images:
        axes = get_iter_axes(images, exclude=EXCLUSION_LIST)
        images.iter_axes = axes
        logger.info(f'Image axes: {axes}')
        for iteration, frame in enumerate(images):

            # Scaling image, needed because image is in 16bit and png has max 256.
            frame_scaled = cv2.normalize(frame, dst=None, alpha=0, beta=65535,
                                         norm_type=cv2.NORM_MINMAX)

            # Fetches metadata info, FOV, time, Z pos, channels
            axes_info = key_value_string(frame.metadata['coords'])

            outfile = f'{args.out_prefix}.{axes_info}.png'
            logger.info(f'Wrinting file {outfile}')
            cv2.imwrite(outfile, frame_scaled)


def key_value_string(dictionary, major_separator = '.', minor_separator='-'):
    string = str()
    for key, value in dictionary.items():
        string += major_separator + str(key) + minor_separator + str(value)
    string = string.lstrip('.')
    return string


def get_iter_axes(nd2reader_parser_object, exclude):

    all_axes_dict = nd2reader_parser_object.sizes.copy()
    axes_dict = delete_keys(all_axes_dict, exclude)
    axes_string = ''.join(axes_dict.keys())

    return axes_string


def delete_keys(dictionary, key_list):
    for key in key_list:
        del dictionary[key]
    return dictionary


# TODO: Make generator
# def nd2_generator(img_path):
#    with ND2Reader(img_path) as images:
#        yield images
#


# TODO: Make it possible to read/write from/to stdin/stdout
def add_arguments(parser):
    parser.add_argument(
        "nd2_image", type=pathlib.Path,
        help="Input image"
    )
    parser.add_argument(
        "out_prefix",
        help="Output prefix. Final name will be <out_prefix>.<iter_axes>-n.png"
    )
