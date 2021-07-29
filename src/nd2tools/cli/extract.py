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
    # parser.add_argument(
    #    "-i", "--iteration", type=int,
    #    help="Choose frame based on capture time."
    # )
    parser.add_argument(
        "--info", action="store_true",
        help="Display possible values for -t, -v and -z for image. Exits after."
    )
    parser.add_argument(
        "-o", "--output",
        help="Write to PNG file."
    )
    parser.add_argument(
        "-m", "--metadata", action="store_true",
        help="Add metadata to output file name. Final name will be "
             "<output>.<iter_axes>-n.png. Default: %(default)s"
    )


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

        if args.metadata:
            metadata = f"t-{args.time}.z-{args.z_pos}.z-{args.FOV}"
        else:
            metadata = False

        print(f"meta: {metadata}")
        output = generate_filename(raw_name=args.output, metadata=metadata)

        logger.info(f"Writing file: {output}")
        cv2.imwrite(output, frame_scaled)


def generate_filename(raw_name, metadata=False):
    name, format = adjust_for_file_extension(raw_name)

    if metadata:
        output = f"{name}.{metadata}.{format}"
    else:
        output = f"{name}.{format}"

    return output


def adjust_for_file_extension(filename, default_format="png",
                              accepted_extensions=("png", "PNG")):
    # No extension
    if "." not in filename:
        return filename, default_format

    # Check extension is accepted (otherwise adds default)
    name, format = filename.rsplit(".", 1)
    if format not in accepted_extensions:
        return filename, default_format
    else:
        return name, format


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
