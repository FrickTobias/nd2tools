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

        if args.info:
            axes_dict = get_iter_axes_dict(images, exclude=EXCLUSION_LIST)
            print_dict(axes_dict, value_descriptor="max+1")
            sys.exit()

        frame = images.get_frame_2D(t=args.time, z=args.z_pos, v=args.FOV)

        # Scaling image, needed because image is in 16bit and png has max 256.
        frame_scaled = cv2.normalize(frame, dst=None, alpha=0, beta=65535,
                                     norm_type=cv2.NORM_MINMAX)

        if not args.name:
            name = ".".join(
                ["t", str(args.time), "z", str(args.z_pos), "v", str(args.FOV)])
        else:
            name = args.name

        cv2.imshow(name, frame_scaled)
        logger.info(f"Displaying image: {name}")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def get_iter_axes_dict(nd2reader_parser_object, exclude):
    all_axes_dict = nd2reader_parser_object.sizes.copy()
    axes_dict = delete_keys(all_axes_dict, exclude)
    return axes_dict


def print_dict(dictionary, key_descriptor="key", value_descriptor="value"):
    for key, val in dictionary.items():
        print(f"{key_descriptor}: {key}\t{value_descriptor}: {val}")


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
        "-t", "--time", default=0, type=int,
        help="Specify time. Default: %(default)s"
    )
    parser.add_argument(
        "-v", "--FOV", default=0, type=int,
        help="Specify field of view. Default: %(default)s"
    )
    parser.add_argument(
        "-z", "--z_pos", default=0, type=int,
        help="Specify Z position. Default: %(default)s"
    )
    parser.add_argument(
        "--info", action="store_true",
        help="Display possible values for -t, -v and -z for image. Exits after."
    )
    parser.add_argument(
        "--name",
        help="Name the displayed image. Default: <Axes information>"

    )
