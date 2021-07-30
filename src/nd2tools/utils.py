import re
from pathlib import Path
import sys
import numpy as np
from collections import Counter

if sys.stderr.isatty():
    from tqdm import tqdm
else:
    def tqdm(iterable, **kwargs):
        return iterable


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
# https://stackoverflow.com/questions/25485886/how-to-convert-a-16-bit-to-an-8-bit-image-in-opencv
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
        # Never changing
        self.original_x1 = x1
        self.original_x2 = x2
        self.original_y1 = y1
        self.original_y2 = y2

        # Cartesian coordinates
        self.x1 = self.original_x1
        self.x2 = self.original_x2
        self.y1 = self.original_y1
        self.y2 = self.original_y2

        # Numpy conversions
        self._update_numpy_coorinates()

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
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self._update_numpy_coorinates()

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
        self.x1 += left
        self.x2 -= right
        self.y1 += bottom
        self.y2 -= top
        self._update_numpy_coorinates()

    # TODO: Add possiblity to keep x/y
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
        x_chunk = int(self.width() / x_pieces)
        y_chunk = int(self.height() / y_pieces)
        self.x1 += x_keep * x_chunk
        self.x2 = self.x1 + x_chunk
        self.y1 += y_keep * y_chunk
        self.y2 = self.y1 + y_chunk
        self._update_numpy_coorinates()

    def width(self):
        return self.x2 - self.x1

    def height(self):
        return self.y2 - self.y1

    def _update_numpy_coorinates(self):
        """
        Run after making any modifications to self.x1, self.x2, self.y1 or self.y2

        Numpy has an inverted y-scale compared to cartesian coordinates, meaning a cut
        starting at (x,y) = (0,0) cuts the top right corner instead of the bottom left.

        Cartesian               Numpy

           ^                       |      x
         y |                    -- + ------->
           |  IMAGE                |
           |                       |  IMAGE
        -- + ------->            y |
           |       x               v

        """
        self.np_y1 = self.original_y2 - self.y2
        self.np_y2 = self.original_y2 - self.y1
        self.np_x1 = self.x1
        self.np_x2 = self.x2
