import re
import sys
import numpy as np
import logging
import time
from collections import Counter
from pathlib import Path

if sys.stderr.isatty():
    from tqdm import tqdm
else:
    def tqdm(iterable, **kwargs):
        return iterable

logger = logging.getLogger(__name__)


def add_global_args(parser):
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
        "--keep", nargs=2, metavar=("X_PIECE", "Y_PIECE"), default=[1, 1],
        help="Specify which piece to keep. Use 0 to keep all and save to "
             "OUTPUT.frame-N.mp4."
    )

    overlay_options = parser.add_argument_group("overlay options")
    overlay_options.add_argument(
        "--scalebar", action="store_true",
        help="Add scalebar to image(s)"
    )
    overlay_options.add_argument(
        "--scalebar-length", type=int, metavar="um",
        help="Length of scalebar."
    )
    overlay_options.add_argument(
        "--timestamps", action="store_true",
        help="Add timestamp to image(s)"
    )


def add_clipping_options(parser):
    clipping_options = parser.add_argument_group("clipping arguments")
    clipping_options.add_argument(
        "--clip-start", type=int, metavar="N", default=0,
        help="Remove N images from start."
    )
    clipping_options.add_argument(
        "--clip-end", type=int, metavar="N", default=0,
        help="Remove N images from end."
    )


def is_1_2(s, t):
    """
    Determine whether s and t are identical except for a single character of
    which one of them is '1' and the other is '2'.
    """
    differences = 0
    one_two = {"1", "2"}
    for c1, c2 in zip(s, t):
        if c1 != c2:
            differences += 1
            if differences == 2:
                return False
            if {c1, c2} != one_two:
                return False
    return differences == 1


def guess_paired_path(path: Path):
    """
    Given the path to a file that contains the sequences for the first read in a
    pair, return the file that contains the sequences for the second read in a
    pair. Both files must have identical names, except that the first must have
    a '1' in its name, and the second must have a '2' at the same position.
    Return None if no second file was found or if there are too many candidates.
    >>> guess_paired_path(Path('file.1.fastq.gz'))  # doctest: +SKIP
    'file.2.fastq.gz'  # if that file exists
    """
    name = path.name
    # All lone 1 digits replaced with '?'
    name_with_globs = re.sub(r"(?<![0-9])1(?![0-9])", "?", name)
    paths = [p for p in path.parent.glob(name_with_globs) if is_1_2(str(p), str(path))]
    if len(paths) == 1:
        return paths[0]
    return None


#
# Modified from stackoverflow
# https://stackoverflow.com/questions/25485886/
# how-to-convert-a-16-bit-to-an-8-bit-image-in-opencv
def map_uint16_to_uint8(img, lower_bound=None, upper_bound=None):
    '''
    Map a 16-bit image trough a lookup table to convert it to 8-bit.

    lower_bound and upper_bound are not 0 and 2^16-1 as default because most of the time
    images do not cover all of the color space, meaning you can retain more information
    by only including part of the 16bit spectra and the linearly convert to 8bit space.
    Here default is min/max of image which is commonly used, for instance by ImageJ.

    Parameters
    ----------
    img: numpy.ndarray[np.uint16]
        image that should be mapped
    lower_bound: int, optional
        lower bound of the range that should be mapped to ``[0, 255]``,
        value must be in the range ``[0, 65535]`` and smaller than `upper_bound`
        (defaults to ``numpy.min(img)``)
    upper_bound: int, optional
       upper bound of the range that should be mapped to ``[0, 255]``,
       value must be in the range ``[0, 65535]`` and larger than `lower_bound`
       (defaults to ``numpy.max(img)``)

    Returns
    -------
    numpy.ndarray[uint8]
    '''
    if lower_bound is None:
        lower_bound = np.min(img)
    if upper_bound is None:
        upper_bound = np.max(img)

    lut = np.concatenate([
        np.zeros(lower_bound, dtype=np.uint16),
        np.linspace(0, 255, upper_bound - lower_bound).astype(np.uint16),
        np.ones(2 ** 16 - upper_bound, dtype=np.uint16) * 255
    ])
    return lut[img].astype(np.uint8)


# TODO: Change so no metadata is empty string (now using None)
def generate_filename(raw_name, metadata=False, format="mp4"):
    name, extension = adjust_for_file_extension(raw_name, format)

    if metadata:
        output = f"{name}.{metadata}.{extension}"
    else:
        output = f"{name}.{extension}"

    return output


def adjust_for_file_extension(filename, format="mp4"):
    # TODO: Change this to standardised method for output path names
    filename = str(filename)

    # No extension
    if "." not in filename:
        return filename, format

    # Check extension is accepted (otherwise adds default)
    # TODO: Change to match statement
    name, extension = filename.rsplit(".", 1)
    if extension != format:
        return filename, format
    else:
        return name, extension


def get_screen_dpi():
    # Get dpi
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    screen = app.screens()[0]
    dpi = screen.physicalDotsPerInch()
    app.quit()
    return dpi


def nd2_get_time(images, format_string="%H:%M:%S"):
    """
    Extracts time information from nd2 images.

    :param images: ImageSequences
    :return t_tring_list: [t1, t2, t3...]
    """
    t_milliseconds_list = images.timesteps
    t_string_list = list()
    for t_milliseconds in t_milliseconds_list:
        t = t_milliseconds / 1000
        t_instance = time.gmtime(t)
        t_string = time.strftime(format_string, t_instance)
        t_string_list.append(t_string)

    return t_string_list


class Summary(Counter):

    def print_stats(self, name=None, value_width=15, print_to=sys.stderr):
        """
        Prints stats in nice table with two column for the key and value pairs in
        summary
        :param name: name of script for header e.g. '__name__'
        :param value_width: width for values column in table
        :param print_to: Where to direct output. Default: stderr
        """
        # Get widths for formatting
        max_name_width = max(map(len, self.keys()), default=10)
        width = value_width + max_name_width + 1

        # Header
        print("=" * width, file=print_to)
        print(f"STATS SUMMARY - {name}", file=print_to)
        print("-" * width, file=print_to)

        # Print stats in columns
        for name, value in self.items():
            value_str = str(value)
            if type(value) is int:
                value_str = f"{value:>{value_width},}"
            elif type(value) is float:
                value_str = f"{value:>{value_width + 4},.3f}"

            print(f"{name:<{max_name_width}} {value_str}", file=print_to)
        print("=" * width, file=print_to)


class ImageCoordinates:
    # TODO: Update comments

    """
    x1,y2 ----------- x2,y2
      |                 |
      |                 |
      |                 |
      |                 |
      |                 |
    x1,y1 ----------- x2,y1
    """

    def __init__(self, x1, x2, y1, y2):
        self.original_x = range(x1, x2 + 1, x2 - x1)
        self.original_y = range(y1, y2 + 1, y2 - y1)
        self.x = self.original_x
        self.y = self.original_y

    def x1(self):
        return self.x[0]

    def x2(self):
        return self.x[-1]

    def y1(self):
        return self.y[0]

    def y2(self):
        return self.y[-1]

    def frame_width(self):
        return self.x[1] - self.x[0]

    def frame_height(self):
        return self.y[1] - self.y[0]

    def cut_out(self, x1, x2, y1, y2):
        """
        Cuts out a part of frame

         + ------- + --------------- + ------- +
         |         |                 |         |
         |         |                 |         |
         |         |                 |         |
         + ----- x1,y2 ----------- x2,y2 ----- +
         |         |                 |         |
         |         |                 |         |
         |         |                 |         |
         |         |                 |         |
         |         |                 |         |
         + ----- x1,y1 ----------- x2,y1 ----- +
         |         |                 |         |
         |         |                 |         |
         |         |                 |         |
         + ------- + --------------- + ------- +

        """

        self._set_xy(x1, x2, y1, y2)

    def trim(self, left, right, bottom, top):
        """
        Removes pixels from left, right, bottom and top of frame

             left                       right
         +----'----+                 +----'----+
         '         '                 '         '

         + ------- + --------------- + ------- +     -+
         |         |                 |         |      |
         |         |                 |         |      } top
         |         |                 |         |      |
         + ----- x1,y2 ----------- x2,y2 ----- +     -+
         |         |                 |         |
         |         |                 |         |
         |         |                 |         |
         |         |                 |         |
         |         |                 |         |
         + ----- x1,y1 ----------- x2,y1 ----- +     -+
         |         |                 |         |      |
         |         |                 |         |       } bottom
         |         |                 |         |      |
         + ------- + --------------- + ------- +     -+

        """

        x1 = self.x1() + left
        x2 = self.x2() - right
        y1 = self.y1() + bottom
        y2 = self.y2() - top

        self._set_xy(x1, x2, y1, y2)

    def split(self, x_pieces, y_pieces, x_keep=0, y_keep=0):
        """
        Splits frame into fractions and keeps one of them

              x_keep = 0
            (x_pieces = 1)
         +--------''--------+
         '                  '

         + ---------------- + ---------------- +
         |                  |                  |
         |                  |                  |
         |                  |                  |
         |                  |                  |
         |                  |                  |
         |                  |                  |
       x1,y2 ------------ x2,y2 -------------- +     -+
         |                  |                  |      |
         |                  |                  |      |
         |                  |                  |       } y_keep = 0 (y_pieces = 2)
         |                  |                  |      |
         |                  |                  |      |
         |                  |                  |      |
       x1,y1 ------------ x2,y1 -------------- +     -+

        """

        x1 = self.x1()
        x2 = self.x2()
        y1 = self.x1()
        y2 = self.x2()

        y_chunk = int(self.frame_height() / y_pieces)
        x_chunk = int(self.frame_width() / x_pieces)

        self._set_xy(x1, x2, y1, y2, x_chunk, y_chunk, x_keep, y_keep)

    def frames(self):
        """
        [(x1, x2, y1, y2), ...]
        """
        frames = list()
        for j, x1 in enumerate(self.x[:-1]):
            for i, y1 in enumerate(self.y[:-1]):
                x_tuple = x1, self.x[j + 1]
                y_tuple = y1, self.y[i + 1]
                frame = x_tuple + y_tuple
                frames.append(frame)
        return frames

    def adjust_frame(self, split, keep, cut, trim):
        """
        Adjusts xy coordinates of image.
        :param self: ImageCoordinates instance
        :param split: Split image into N pieces
        :param keep: Which piece after after split
        :param cut: Cut image to (x1, x2, y1, y2)
        :param trim: Trim (left, right, top, bottom) pixels from image
        """
        if split:
            keep_0_index = [int(xy) - 1 for xy in keep]
            x, y = keep_0_index
            self.split(*split, x_keep=x, y_keep=y)
            logger.info(
                f"Slicing image into a {split} grid"
            )
        if cut:
            self.cut_out(*cut)
            logger.info(
                f"Cutting frames to: x={self.x1()}:{self.x2()}, "
                f"y={self.y1()}:{self.y2()}"
            )
        if trim:
            self.trim(*trim)
            logger.info(
                f"Trimming frames to: x={self.x1()}:{self.x2()}, "
                f"y={self.y1()}:{self.y2()}"
            )

    def _set_xy(self, x1, x2, y1, y2, x_chunk=None, y_chunk=None, x_keep=0, y_keep=0):

        # x
        if not x_chunk:
            x_chunk = x2 - x1
        if x_keep <= -1:
            self.x = range(x1, x2 + 1, x_chunk)
        else:
            self.x = range(x1, x2 + 1, x_chunk)[x_keep:x_keep + 2]

        # y
        if not y_chunk:
            y_chunk = y2 - y1
        if y_keep <= -1:
            self.y = range(y1, y2 + 1, y_chunk)
        else:
            self.y = range(y1, y2 + 1, y_chunk)[y_keep:y_keep + 2]


class ScalingMinMax:
    """
    Tracks min/max values for 16bit to 8bit conversion.
    """

    def __init__(self, mode, image, scaling=0, ):
        """
        Usage:

            self.min_current
            self.max_current
            self.update(image)

        :param mode: [first, continuous, current, naive]
        :param scaling: Widens min/max. For 0.2 min = min * 0.8 and max = max * 1.2
        :param images: PIMS image
        """

        self.min_current = 65535  # Just needs to be >= first value so it gets updated
        self.max_current = int()  # Just needs to be <= first value so it gets updated

        self.min_updates = int()
        self.max_updates = int()
        self._min_scaling = 1 - scaling
        self._max_scaling = 1 + scaling

        self.mode = mode
        if self.mode == "first" or self.mode == "continuous" or self.mode == "current":
            min_init = np.min(image)
            max_init = np.max(image)
        elif self.mode == "naive":
            min_init = 0
            max_init = 65535  # = 16^2 - 1
        else:
            raise AttributeError(self.mode)

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
