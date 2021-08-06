"""
Converts images to MP4 movies
"""

import pathlib
import logging
import cv2
import numpy as np
from tqdm import tqdm
from nd2reader import ND2Reader

from nd2tools.utils import ImageCoordinates
from nd2tools.utils import map_uint16_to_uint8
from nd2tools.utils import generate_filename

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

logger = logging.getLogger(__name__)


def add_arguments(parser):
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

    clipping_options = parser.add_argument_group("clipping arguments")
    clipping_options.add_argument(
        "--clip-start", type=int, metavar="N", default=0,
        help="Remove N images from start."
    )
    clipping_options.add_argument(
        "--clip-stop", type=int, metavar="N", default=0,
        help="Remove N images from end."
    )

    cropping_options = parser.add_argument_group("cropping arguments")
    cropping_options.add_argument(
        "--cut", nargs=4, type=int,
        metavar=("X1", "X2", "Y1", "Y2"),
        help="Cut out rectangle defined by x1,y1 and x2,y2. By numpy convention 0,0 is "
             "top left pixel and y axis points down"
    )
    cropping_options.add_argument(
        "--trim", nargs=4, type=int,
        metavar=("LEFT", "RIGHT", "TOP", "BOTTOM"),
        help="Trim images [pixels]."
    )
    cropping_options.add_argument(
        "--split", type=int, nargs=2, metavar=("X_PIECES", "Y_PIECES"),
        help="Split images into a X_PIECES by Y_PIECES grid. See --keep for which "
             "piece(s) to save."
    )
    cropping_options.add_argument(
        "--keep", nargs=2, default=["1", "1"], metavar=("X_PIECE", "Y_PIECE"),
        help="Specify which piece to keep. Use 0 to keep all and save to "
             "OUTPUT.frame-N.mp4. %(default)s."
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
        im_xy = adjust_frame(im_xy, args.split, args.keep, args.cut, args.trim)

        frame_pos_list = im_xy.frames()
        write_video_greyscale(file_prefix=args.output, images=images, fps=args.fps,
                              width=im_xy.frame_width(), height=im_xy.frame_height(),
                              frame_pos_list=frame_pos_list,
                              conversion_method=args.conversion,
                              scale_conversion=args.scale_conversion,
                              clip_start=args.clip_start, clip_stop=args.clip_stop)
        logger.info("Finished")


def adjust_frame(image_coordinates, split, keep, cut, trim):
    """
    Adjusts xy coordinates of image.
    :param image_coordinates: ImageCoordinates instance
    :param split: Split image into N pieces
    :param keep: Which piece after after split
    :param cut: Cut image to (x1, x2, y1, y2)
    :param trim: Trim (left, right, top, bottom) pixels from image
    """
    if split:
        keep_0_index = [int(xy) - 1 for xy in keep]
        x, y = keep_0_index
        image_coordinates.split(*split, x_keep=x, y_keep=y)
        logger.info(
            f"Slicing image into a {split} grid"
        )
    if cut:
        image_coordinates.cut_out(*cut)
        logger.info(
            f"Cutting frames to: x={image_coordinates.x1()}:{image_coordinates.x2()}, "
            f"y={image_coordinates.y1()}:{image_coordinates.y2()}"
        )
    if trim:
        image_coordinates.trim(*trim)
        logger.info(
            f"Trimming frames to: x={image_coordinates.x1()}:{image_coordinates.x2()}, "
            f"y={image_coordinates.y1()}:{image_coordinates.y2()}"
        )

    return image_coordinates


def write_video_greyscale(file_prefix, images, fps, width, height,
                          frame_pos_list, conversion_method="first", scale_conversion=0,
                          clip_start=0, clip_stop=0):
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
    """

    # Opens one file per frame_pos tracks using open_video_files.dict[frame_pos] = writer
    open_video_files = OpenVideoFiles(file_prefix, fps, width, height, frame_pos_list)

    # Writing outputs
    scaling_min_max = ScalingMinMax(mode=conversion_method, scaling=scale_conversion,
                                    images=images)
    first_frame = clip_start
    last_frame = len(images) - clip_stop
    for frame_number, image in enumerate(
            tqdm(images, desc=f"Writing movie file(s)", unit=" images",
                 total=last_frame)):

        if frame_number < first_frame:
            continue

        # Split image and writes to appropriate files
        for frame_pos in frame_pos_list:
            x1, x2, y1, y2 = frame_pos
            image_cropped = image[y1:y2, x1:x2]

            if scaling_min_max.mode == "continuous" or scaling_min_max.mode == "current":
                logger.info(f"frame: {frame_number}")
                scaling_min_max.update(image_cropped)

            image_cropped_8bit = map_uint16_to_uint8(image_cropped,
                                                     lower_bound=scaling_min_max.min_current,
                                                     upper_bound=scaling_min_max.max_current)

            # TODO: do this properly
            time = get_metadata(image)
            print(time)
            image_cropped_8bit = add_text_to_image(image_cropped_8bit, text="TESTING!")

            open_video_files.dictionary[frame_pos].write(image_cropped_8bit)

        if frame_number >= last_frame:
            break

    # Close files
    open_video_files.close()

# TODO: Dot this properly
def get_metadata(image):
    time = image.metadata["date"]
    import pdb
    pdb.set_trace()
    return time

# TODO: Dot this properly
def add_text_to_image(image, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    pos = (100, 100)
    cv2.putText(image, text, pos, font, fontScale=3, color=(255, 255, 255), thickness=5)
    cv2.imshow("image", image)
    return image

    #return image


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


class ScalingMinMax:
    """
    Tracks min/max values for 16bit to 8bit conversion.
    """

    def __init__(self, mode, scaling, images):
        """
        Usage:

            self.min_current
            self.max_current
            self.update(image)

        :param mode: [first, continuous, current, naive]
        :param scaling: Widens min/max. For 0.2 min = min * 0.8 and max = max * 1.2
        :param images: PIMS image
        """

        self.min_current = 65535
        self.max_current = int()

        self.min_updates = int()
        self.max_updates = int()
        self._min_scaling = 1 - scaling
        self._max_scaling = 1 + scaling

        self.mode = mode
        if self.mode == "first" or self.mode == "continuous" or self.mode == "current":
            min_init = np.min(images[0])
            max_init = np.max(images[0])
        elif self.mode == "naive":
            min_init = 0
            max_init = 65535  # = 16^2 - 1

        self._set_min(min_init)
        self._set_max(max_init)

    def update(self, image):

        image_min = np.min(image[0])
        image_max = np.max(image[0])

        if self.mode == "current":
            self._set_min(image_min)
            self._set_max(image_max)

        elif self.mode == "continuous":
            self._keep_lower(image_min)
            self._keep_higher(image_max)

    def _keep_lower(self, min_new):
        if min_new < self.min_current:
            self._set_min(min_new)

    def _keep_higher(self, max_new):
        if max_new > self.max_current:
            self._set_max(max_new)

    def _set_min(self, min_new):

        self.raw_min = min_new
        self.min_current = int(self.raw_min * self._min_scaling)

        logger.info(
            f"16 bit min = {self.min_current}. Updated {self.min_updates} time(s).")
        self.min_updates += 1

    def _set_max(self, max_new):

        self.raw_max = max_new
        self.max_current = int(self.raw_max * self._max_scaling)

        logger.info(
            f"16 bit max = {self.max_current}. Updated {self.max_updates} time(s).")
        self.max_updates += 1
