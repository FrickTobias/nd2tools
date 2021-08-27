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
from nd2tools.utils import cv2_add_text_to_image
from nd2tools.utils import Cv2ImageText

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


def image(input, output, split=None, keep=None, cut=None, trim=None):
    with ND2Reader(input) as images:

        # import pdb
        # pdb.set_trace()

        im_xy = ImageCoordinates(x1=0, x2=images.sizes['x'], y1=0, y2=images.sizes['y'])
        im_xy.adjust_frame(split, keep, cut, trim)
        frame_pos_list = im_xy.frames()
        scaling_min_max = ScalingMinMax(mode="continuous",
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
                        scaling_min_max.update(image_crop)
                    image_crop = map_uint16_to_uint8(image_crop,
                                                     lower_bound=scaling_min_max.min_current,
                                                     upper_bound=scaling_min_max.max_current)

                image_crop = cv2_add_scalebar(image_crop, px_size=image.metadata["pixel_microns"])

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


def cv2_add_scalebar(image, px_size, pos=(50, 50), color=(0, 0, 0)):
    """
    """

    # Get variables
    im_height, im_width = image.shape

    # Add box to image
    box_end_pos, real_length = cv2_scalebar_end_pos(pos, px_size=px_size, im_height=im_height,
                                       im_width=im_width)
    cv2.rectangle(image, pos, box_end_pos, color, -1)


    # Add text
    text = f"{real_length} um"
    img_txt = Cv2ImageText(pos=pos) # TODO: set defaults in cv2_add_text_to_image and remove
    box_x1, box_y1 = pos
    box_x2, box_y2 = box_end_pos
    separator = 10
    text_dim, _ = cv2.getTextSize(text, img_txt.font, img_txt.size, img_txt.thickness)
    _, text_h = text_dim
    txt_pos = (box_x1, box_y2 + text_h + img_txt.size - 1 + separator)
    image = cv2_add_text_to_image(image, text,
                                       img_txt.font, img_txt.size,
                                       img_txt.thickness, txt_pos,
                                       img_txt.color_cv2)
    return image


def cv2_scalebar_end_pos(pos, px_size, im_height, im_width, height_frac=0.01,
                         width_frac=0.1):
    """
    """

    # calculate len
    delta_y = int(im_height * height_frac)
    # calculate width
    delta_x = int(im_width * width_frac)

    x_size = delta_x * px_size

    x, y = pos
    end_pos = (x + delta_x, y + delta_y)

    return end_pos, x_size
