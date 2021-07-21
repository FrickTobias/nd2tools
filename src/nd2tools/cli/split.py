"""
Splits nd2 images into individual images.
"""

import pathlib
import logging
import cv2
import sys
from nd2reader import ND2Reader

logger = logging.getLogger(__name__)
EXCLUSION_LIST = list('x' 'y')


def main(args):
    with ND2Reader(args.input) as images:
        axes = get_iter_axes(images, exclude=EXCLUSION_LIST)
        images.iter_axes = axes
        logger.info(f'Image axes: {axes}')
        for iteration, frame in enumerate(images):
            # Scaling image, needed because image is in 16bit and png has max 256.
            frame_scaled = cv2.normalize(frame, dst=None, alpha=0, beta=65535,
                                         norm_type=cv2.NORM_MINMAX)

            # Fetches metadata info, FOV, time, Z pos, channels
            axes_info = key_value_string(frame.metadata['coords'])
            output = f'{args.output}.{axes_info}.png'
            logger.info(f'Writing to file {output}')
            cv2.imwrite(output, frame_scaled)


def key_value_string(dictionary, major_separator='.', minor_separator='-'):
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


def add_arguments(parser):
    parser.add_argument(
        "input", type=pathlib.Path,
        help="Input PNG image"
    )
    parser.add_argument(
        "-o", "--output",
        help="Write to PNG file(s). Final name will be <output>.<iter_axes>-n.png."
    )
