"""
Adds scalebars to images
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

logger = logging.getLogger(__name__)


# TODO: Change so it assumes one image at the time and reads .png
# TODO: Aka adapt to nd2tools split
def add_arguments(parser):
    parser.add_argument(
        "input", type=pathlib.Path,
        help="Image to add scale bar to."
    )
    parser.add_argument(
        "output",
        help="Output file name. Will save in PNG."
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


def main(args):
    scalebar(input=args.input, output=args.output, split=args.split, keep=args.keep,
             cut=args.cut, trim=args.trim)


def scalebar(input, output, split, keep, cut, trim):
    with ND2Reader(input) as images:
        im_xy = ImageCoordinates(x1=0, x2=images.sizes['x'], y1=0, y2=images.sizes['y'])
        im_xy = adjust_frame(im_xy, split, keep, cut, trim)
        frame_pos_list = im_xy.frames()
        scaling_min_max = ScalingMinMax(mode="continuous",
                                        scaling=1,
                                        images=images)

        for frame_number, image in enumerate(images):

            # ims = list()
            for frame_fraction, frame_pos in enumerate(frame_pos_list):

                # Crop image
                x1, x2, y1, y2 = frame_pos
                image_crop = image[y1:y2, x1:x2]

                # convert 16bit to 8bit
                if image_crop.dtype == "uint16":
                    if scaling_min_max.mode == "continuous" or scaling_min_max.mode == "current":
                        logger.info(f"frame: {frame_number}")
                        scaling_min_max.update(image_crop)
                    image_crop = map_uint16_to_uint8(image_crop,
                                                     lower_bound=scaling_min_max.min_current,
                                                     upper_bound=scaling_min_max.max_current)

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


def get_screen_dpi():
    # Get dpi
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    screen = app.screens()[0]
    dpi = screen.physicalDotsPerInch()
    app.quit()
    return dpi


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
