"""
Writes mp4 videos from nd2 files
"""

import pathlib
import logging
import cv2
import sys
import matplotlib
from tqdm import tqdm
from PyQt5.QtWidgets import QApplication
from nd2reader import ND2Reader
from nd2reader import raw_metadata

from nd2tools.utils import map_uint16_to_uint8
from nd2tools.utils import generate_filename
from nd2tools.utils import add_global_args
from nd2tools.utils import get_screen_dpi

from nd2tools.utils import cv2_gray_to_color
from nd2tools.utils import cv2_add_text_to_image
from nd2tools.utils import cv2_build_overlay
from nd2tools.utils import cv2_remove_white_background
from nd2tools.utils import plt_to_cv2
from nd2tools.utils import nd2_get_time
from nd2tools.utils import add_clipping_options

from nd2tools.utils import ImageCoordinates
from nd2tools.utils import ScalingMinMax
from nd2tools.utils import Cv2ImageText

logger = logging.getLogger(__name__)

#
# Non-GUI matplotlib backend for solving AttributeError: 'FigureCanvasMac' object has no attribute 'renderer'
# Comment out to use plt.show()
matplotlib.use('agg')


def add_arguments(parser):
    add_global_args(parser)
    add_clipping_options(parser)
    parser.add_argument(
        "input", type=pathlib.Path,
        help="Input PNG image"
    )
    parser.add_argument(
        "output",
        help="Output MP4 file."
    )
    parser.add_argument(
        "-f", "--fps", type=int, default=30,
        help="Frames per second. Default: %(default)s"
    )

    overlay_options = parser.add_argument_group("overlay options")
    overlay_options.add_argument(
        "-s", "--scalebar", action="store_true",
        help="Add scalebar to image. See --magnification to set objective magnification."
    )
    overlay_options.add_argument(
        "--magnification", type=float, default=10,
        help="Objective magnification used at image acquisition. %(default)s"
    )

    conversion_options = parser.add_argument_group("bit conversion arguments")
    conversion_options.add_argument(
        "--conversion", choices=["first", "continuous", "current", "naive"],
        default="first",
        help="Determines how min/max are set for conversion from 16bit to 8bit color "
             "space. first: first_image. continuous: images_read(image_number). "
             "current: current_image. naive: Uses min=0 and max=2^16-1. Default: "
             "%(default)s."
    )
    conversion_options.add_argument(
        "--scale-conversion", type=float, default=0,
        help="Scale min/max interval for 16bit/8bit conversion. Formula: scaled_max = "
             "max * (1 + k) and scalad_min = min * (1 - k). Default: %(default)s."
    )


def main(args):
    with ND2Reader(args.input) as images:
        # Adjusting output frame coordinates
        im_xy = ImageCoordinates(x1=0, x2=images.sizes['x'], y1=0, y2=images.sizes['y'])
        im_xy.adjust_frame(args.split, args.keep, args.cut, args.trim)

        frame_pos_list = im_xy.frames()
        movie(images=images, output=args.output, fps=args.fps,
              width=im_xy.frame_width(), height=im_xy.frame_height(),
              frame_pos_list=frame_pos_list,
              conversion_method=args.conversion,
              scale_conversion=args.scale_conversion,
              clip_start=args.clip_start, clip_end=args.clip_end,
              scalebar=args.scalebar, magnification=args.magnification)
        logger.info("Finished")


def movie(images, output, fps, width, height, frame_pos_list, conversion_method="first",
          scale_conversion=0, clip_start=0, clip_end=0, scalebar=None,
          magnification=10):
    """
    Writes images to an mp4 video file
    :param file_path: Path to output video, must end with .mp4
    :param images: List of PIL.Image objects
    :param fps: Desired frame rate
    :param width: Width of images
    :param height: Height of images
    :param frame_pos_list: List of tuples, [(x1, x2, y1, y2), ...] for image cropping
    :param conversion_method: 16bit to 8bit color conversion method
    :param scale_conversion: Factor to widen min/max for color conversion (0.2 => *1.2/0.8)
    :param clip_start: Start frame number
    :param clip_stop: Stop frame number
    :param magnification: Objective magnification from image acquisition
    """
    # TODO: Move PROCESSSING out of this function

    # Opens one file per frame_pos tracks using open_video_files.dict[frame_pos] = writer
    open_video_files = OpenVideoFiles(output, fps, width, height, frame_pos_list,
                                      is_color=1)

    pixel_size = images.metadata["pixel_microns"]
    img_txt = Cv2ImageText()
    overlay = cv2_build_overlay(width=width, height=height, scalebar=scalebar,
                                magnification=magnification, pixel_size=pixel_size,
                                color=img_txt.color_matplotlib)
    scaling_min_max = ScalingMinMax(mode=conversion_method, scaling=scale_conversion,
                                    image=images[0])
    first_frame = clip_start
    last_frame = len(images) - clip_end
    timesteps = nd2_get_time(images)
    for frame_number, image in enumerate(
            tqdm(images[first_frame:last_frame], desc=f"Writing movie file(s)",
                 unit=" images",
                 total=last_frame - first_frame)):

        # Split image and writes to appropriate files
        acquisition_time = timesteps[frame_number]
        # ims = list()
        for frame_pos in frame_pos_list:

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

            # Add text (changes for different images)
            image_crop = cv2_gray_to_color(image_crop)
            image_crop = cv2_add_text_to_image(image_crop, f"t: {acquisition_time}",
                                               pos=img_txt.pos, color=img_txt.color_cv2,
                                               background=True)

            # TODO: Change to cv2 scalebar function (and add to overlay function?)

            # Add overlay
            image_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2BGRA)
            image_crop = cv2.addWeighted(image_crop, 1, overlay, 1, 0)
            image_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGRA2BGR)

            # Write image_crop
            open_video_files.dictionary[frame_pos].write(image_crop)

    # Close files
    open_video_files.close()


class OpenVideoFiles:
    """
    Wrapper object for cv2.VideoWriter for opening a generalized number of files for
    writing.
    """

    def __init__(self, file_prefix, fps, width, height, frame_pos_list, is_color=0):
        """
        Usage:

            # frame_tuple = (x1, x2, y1, y2)
            open_video_file.dictionary[frame_tuple].write(outline)

        :param file_prefix: Output name/prefix
        :param fps: Frames per second
        :param width: Width of output videos
        :param height: Height of output videos
        :param frame_pos_list: List of frame coordinate tuples [(x1, x2, y1, y2), ...]
        output name generation and writer tracking
        :param is_color: See cv2.VideoWriter() isColor. 0 Writes in grayscale.
        """

        # Opens one file per frame position set
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.dictionary = dict()
        for count, frame_pos in enumerate(frame_pos_list):

            # If multiple output, add frame-N to file name
            if len(frame_pos_list) >= 2:
                metadata = f"frame-{count + 1}"
            else:
                metadata = False
            file_path = generate_filename(file_prefix, metadata=metadata)

            # Open writers and track open files using dictionary
            logger.info(f"Creating file: {file_path}")
            writer = cv2.VideoWriter(filename=file_path, fourcc=fourcc, fps=fps,
                                     frameSize=(width, height), isColor=is_color)
            self.dictionary[frame_pos] = writer

    def close(self):
        for writer in self.dictionary.values():
            writer.release()
