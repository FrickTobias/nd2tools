"""
Displays images from nd2 files
"""

import pathlib
import logging
import cv2
import sys
from nd2reader import ND2Reader

from nd2tools.utils import map_uint16_to_uint8
from nd2tools.utils import ImageCoordinates
from nd2tools.utils import ScalingMinMax
from nd2tools.utils import add_global_args

logger = logging.getLogger(__name__)
EXCLUSION_LIST = list('x' 'y')


def add_arguments(parser):
    add_global_args(parser)
    parser.add_argument(
        "input", type=pathlib.Path,
        help="Input PNG image"
    )
    parser.add_argument(
        "-t", "--time", type=int,
        help="Specify time."
    )
    parser.add_argument(
        "-v", "--FOV", type=int,
        help="Specify field of view."
    )
    parser.add_argument(
        "-z", "--z_pos", type=int,
        help="Specify Z position."
    )
    parser.add_argument(
        "--choices", action="store_true",
        help="Display possible values for -t, -v, -z and exit."
    )


def main(args):
    if args.choices:
        img_info = get_iter_axes_dict(args.input, exclude=EXCLUSION_LIST)
        print_dict(img_info)
        sys.exit()

    display(input=args.input, split=args.split, keep=args.keep,
            cut=args.cut, trim=args.trim, time=args.time, z_pos=args.z_pos,
            FOV=args.FOV)


def display(input, split=None, keep=None, cut=None, trim=None,
            time=0, z_pos=0, FOV=0, duration=0):
    with ND2Reader(input) as images:

        im_xy = ImageCoordinates(x1=0, x2=images.sizes['x'], y1=0, y2=images.sizes['y'])
        im_xy.adjust_frame(split, keep, cut, trim)
        frame_pos_list = im_xy.frames()
        scaling_min_max = ScalingMinMax(mode="continuous", scaling=1, image=images[0])

        image = images.get_frame_2D(t=time, z=z_pos, v=FOV)
        # ims = list()
        for frame_fraction, frame_pos in enumerate(frame_pos_list):

            # Crop image
            x1, x2, y1, y2 = frame_pos
            image_crop = image[y1:y2, x1:x2]

            # convert 16bit to 8bit
            if image_crop.dtype == "uint16":
                image_crop = map_uint16_to_uint8(image_crop,
                                                 lower_bound=scaling_min_max.min_current,
                                                 upper_bound=scaling_min_max.max_current)

            name = f"t-{time}.z-{z_pos}.v-{FOV}"

            display_image(name, image_crop, duration)

    logger.info("Finished")


def display_image(name, frame, duration=0):
    cv2.imshow(name, frame)
    logger.info(f"Displaying image: {name}")
    logger.info("Push any key to continue.")
    cv2.waitKey(duration)
    logger.info("Closing windows")
    cv2.destroyAllWindows()


def get_iter_axes_dict(nd2file, exclude):
    with ND2Reader(nd2file) as images:
        all_axes_dict = images.sizes.copy()
        axes_dict = delete_keys(all_axes_dict, exclude)
        return axes_dict


def delete_keys(dictionary, key_list):
    for key in key_list:
        del dictionary[key]
    return dictionary


def print_dict(dictionary, key_descriptor="key", value_descriptor="value"):
    for key, val in dictionary.items():
        print(f"{key_descriptor}: {key}\t{value_descriptor}: {list(range(val))}")
