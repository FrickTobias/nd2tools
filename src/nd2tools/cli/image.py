"""
Writes png images from nd2 files
"""

import matplotlib.pyplot as plt
import imageio
import pathlib
import logging
import numpy as np
import cv2
from tqdm import tqdm

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
from nd2tools.utils import add_clipping_options
from nd2tools.utils import cv2_gray_to_color

logger = logging.getLogger(__name__)

SEPARATOR = 10


def add_arguments(parser):
    add_global_args(parser)
    add_clipping_options(parser)
    parser.add_argument(
        "input", type=pathlib.Path,
        help="Nd2 file"
    )
    parser.add_argument(
        "output",
        help="Output file name. Will save in PNG."
    )
    parser.add_argument(
        "--scalebar-length", type=int,
        help="Length of scalebar in micro meters."
    )


def main(args):
    image(input=args.input, output=args.output, clip_start=args.clip_start,
          clip_end=args.clip_end, split=args.split, keep=args.keep,
          cut=args.cut, trim=args.trim, scalebar_length=args.scalebar_length)


def image(input, output, clip_start=0, clip_end=0, split=None, keep=None, cut=None,
          trim=None, scalebar_length=None):
    with ND2Reader(input) as images:
        im_txt = Cv2ImageText()
        im_xy = ImageCoordinates(x1=0, x2=images.sizes['x'], y1=0, y2=images.sizes['y'])
        im_xy.adjust_frame(split, keep, cut, trim)
        frame_pos_list = im_xy.frames()
        scaling_min_max = ScalingMinMax(mode="continuous",
                                        image=images[0])
        pixel_size = images[0].metadata["pixel_microns"]
        first_frame = clip_start
        last_frame = len(images) - clip_end
        assert images[0].dtype == "uint16", f"Not 16bit image ({images[0].dtype})"

        # TODO: Move for_every_image to separate function
        for image_number, image in enumerate(tqdm(images[first_frame:last_frame],
                                                  desc=f"Writing image file(s)",
                                                  unit=" images",
                                                  total=last_frame - first_frame),
                                             start=first_frame):
            if scaling_min_max.mode == "continuous" or scaling_min_max.mode == "current":
                scaling_min_max.update(image)
            image_8bit = map_uint16_to_uint8(image,
                                             lower_bound=scaling_min_max.min_current,
                                             upper_bound=scaling_min_max.max_current)

            # If splitting image, iterate over frames
            for frame_fraction, frame_pos in enumerate(frame_pos_list):
                image_8bit_crop = crop_image(image_8bit, frame_pos)
                image_8bit_crop_color = cv2_gray_to_color(image_8bit_crop)
                image_8bit_crop_color_scalebar = cv2_add_scalebar(image_8bit_crop_color,
                                                                  pixel_size,
                                                                  color=im_txt.color_cv2,
                                                                  length=scalebar_length)

                # Generate filename and write to out
                metadata = build_metadata_string(images, image_number, frame_pos_list,
                                                 frame_fraction)
                file_path = generate_filename(output, metadata=metadata, format="png")
                cv2.imwrite(file_path, image_8bit_crop_color_scalebar)


def crop_image(image, frame_pos):
    x1, x2, y1, y2 = frame_pos
    image_crop = image[y1:y2, x1:x2]
    return image_crop


def build_metadata_string(images, image_number, frame_pos_list, frame_fraction):
    metadata = list()
    if len(images) >= 2:
        metadata.append(f"image-{image_number + 1}")
    if len(frame_pos_list) >= 2:
        metadata.append(f"frame-{frame_fraction + 1}")

    if len(metadata) >= 1:
        metadata = ".".join(metadata)
    else:
        metadata = False

    return metadata


def cv2_add_scalebar(image, px_size, pos=(50, 50), color=(0, 0, 0), length=None):
    """
    """

    # Get variables
    im_height, im_width, _ = image.shape

    # Add box to image
    box_end_pos, real_length = cv2_scalebar_end_pos(pos, px_size=px_size,
                                                    im_height=im_height,
                                                    im_width=im_width, length=length)
    cv2.rectangle(image, pos, box_end_pos, color, -1)

    # String under box
    text = f"{real_length} um"

    # Calculate text position
    img_txt = Cv2ImageText()
    box_x1, box_y1 = pos
    box_x2, box_y2 = box_end_pos
    text_dim, _ = cv2.getTextSize(text, img_txt.font, img_txt.size, img_txt.thickness)
    _, text_h = text_dim
    txt_pos = (box_x1, box_y2 + text_h + img_txt.size - 1 + SEPARATOR)

    # Add text to image
    image = cv2_add_text_to_image(image, text, size=img_txt.size, font=img_txt.font,
                                  color=img_txt.color_cv2, pos=txt_pos)
    return image


def cv2_scalebar_end_pos(pos, px_size, im_height, im_width, height_frac=0.01,
                         width_frac=0.1, length=None, decimals=2):
    """
    """

    # calculate len
    delta_y = int(im_height * height_frac)

    # calculate width
    if length:
        delta_x = round(length / px_size)
    else:
        delta_x = round(im_width * width_frac)
    x_size = round(delta_x * px_size, decimals)

    # Convert to px positions in image
    x, y = pos
    end_pos = (x + delta_x, y + delta_y)

    return end_pos, x_size
