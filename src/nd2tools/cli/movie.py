"""
Converts images to MP4 movies
"""

import pathlib
import logging
import time
import cv2
import sys
import numpy as np
from tqdm import tqdm
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from matplotlib_scalebar.scalebar import ScaleBar
from PyQt5.QtWidgets import QApplication

from nd2reader import ND2Reader

from nd2tools.utils import ImageCoordinates
from nd2tools.utils import map_uint16_to_uint8
from nd2tools.utils import generate_filename

logger = logging.getLogger(__name__)

#
# Non-GUI matplotlib backend for solving AttributeError: 'FigureCanvasMac' object has no attribute 'renderer'
# Comment out to use plt.show()
matplotlib.use('agg')


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

    overlay_options = parser.add_argument_group("overlay options")
    overlay_options.add_argument(
        "-s", "--scalebar", action="store_true",
        help="Add scalebar to image. See --magnification to set objective magnification."
    )
    overlay_options.add_argument(
        "--magnification", type=float, default=10,
        help="Objective magnification used at image acquisition. %(default)s"
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
        write_video(file_prefix=args.output, images=images, fps=args.fps,
                    width=im_xy.frame_width(), height=im_xy.frame_height(),
                    frame_pos_list=frame_pos_list,
                    conversion_method=args.conversion,
                    scale_conversion=args.scale_conversion,
                    clip_start=args.clip_start, clip_stop=args.clip_stop,
                    scalebar=args.scalebar, magnification=args.magnification)
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


def write_video(file_prefix, images, fps, width, height,
                frame_pos_list, conversion_method="first", scale_conversion=0,
                clip_start=0, clip_stop=0, scalebar=None, magnification=10):
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
    open_video_files = OpenVideoFiles(file_prefix, fps, width, height, frame_pos_list,
                                      is_color=1)

    # TODO: Move to better place
    pixel_size = images.metadata["pixel_microns"]

    # CREATE text instance
    img_txt = Cv2ImageText()

    print(img_txt.color_cv2)

    overlay = build_overlay(width=width, height=height, scalebar=scalebar,
                            magnification=magnification, pixel_size=pixel_size,
                            color=img_txt.color_matplotlib)

    # Writing outputs
    scaling_min_max = ScalingMinMax(mode=conversion_method, scaling=scale_conversion,
                                    images=images)
    first_frame = clip_start
    last_frame = len(images) - clip_stop
    timesteps = nd2_get_time(images)
    for frame_number, image in enumerate(
            tqdm(images, desc=f"Writing movie file(s)", unit=" images",
                 total=last_frame)):

        # Option: --clip-start
        # TODO: Change to for i in images[first_image:last_image]
        if frame_number < first_frame:
            continue

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
                    logger.info(f"frame: {frame_number}")
                    scaling_min_max.update(image_crop)
                image_crop = map_uint16_to_uint8(image_crop,
                                                 lower_bound=scaling_min_max.min_current,
                                                 upper_bound=scaling_min_max.max_current)

            # Add text to frames
            image_crop = cv2_gray_to_color(image_crop)
            image_crop = cv2_add_text_to_image(image_crop,
                                               text=f"t: {acquisition_time}",
                                               background=True, img_txt=img_txt)

            image_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2BGRA)

            image_crop = cv2.addWeighted(image_crop, 1, overlay, 1, 0)

            image_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGRA2BGR)

            # Write image_crop
            open_video_files.dictionary[frame_pos].write(image_crop)

        # Option: --clip-end
        if frame_number >= last_frame:
            break

    # Close files
    open_video_files.close()


# TODO: Change scalebar/magnification thing
def build_overlay(width, height, scalebar, magnification, pixel_size, color):
    # magnification = scalebar
    # print(px_microns)
    # import pdb
    # pdb.set_trace()

    overlay_plt, overlay_fig, overlay_ax = add_scalebar(width=width, height=height,
                                                        pixel_size=pixel_size,
                                                        magnification=magnification,
                                                        color=color)

    overlay_image = plt_to_cv2(figure=overlay_fig, width=width, height=height)

    # remove white background
    overlay_image = cv2_remove_white_background(overlay_image)

    return overlay_image


def add_scalebar(width, height, pixel_size, magnification, color):
    # Get screen pixel density
    dpi = get_screen_dpi()
    # Compensate size of pixel for objective magnification
    pixel_size_real = pixel_size * magnification

    # Create subplot. Specifies figsize and dpi in order to keep original resolution
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)

    # Default behaviour in plt is to have space surrounding plot
    ax = fig.add_axes([0, 0, 1, 1])

    # ax.axis("off") will override stretching plot area to fill entire fig space
    _ = [ax.spines[axis].set_visible(False) for axis in ax.spines.keys()]

    scalebar_width = 0.01
    scalebar_length = 0.1
    font = {
        "size": "25"
    }

    # Create scale bar
    scalebar = ScaleBar(pixel_size_real, "um", frameon=False,
                        length_fraction=scalebar_length,
                        width_fraction=scalebar_width, font_properties=font,
                        location="upper right", border_pad=2.5, color=color)
    ax.add_artist(scalebar)

    return plt, fig, ax


def get_screen_dpi():
    app = QApplication(sys.argv)
    screen = app.screens()[0]
    dpi = screen.physicalDotsPerInch()
    app.quit()
    return dpi


def plt_to_cv2(figure, width=None, height=None):
    """
    Convert matplotlib plot to cv2 image

    :param figure: Matplotlib figure (same as fig, _ from matplotlib.pyplot.subplot())
    :param width: Width of output [px]. Defaults to width of figure.
    :param height: Height of output [px]. Defaults to height of figure.
    :return image_cv2: cv2 image
    """

    # TODO: Check through all this and check docs!

    # Init
    figure.canvas.draw()
    if not width or not width:
        width, height = figure.canvas.get_width_height()

    # Get the RGB buffer from the figure
    tmp = figure.canvas.tostring_argb()  # Prep for np conversion
    buf = np.fromstring(tmp, dtype=np.uint8)  # Store in np array
    buf.shape = (height, width, 4)  # Set shape of array
    buf = np.roll(buf, 3, axis=2)  # ARGB => RGBA

    # Convert to cv2 image
    image_cv2 = Image.frombytes("RGBA", (width, height),
                                buf.tostring())  # Make cv2 image
    image_cv2.convert("RGBA")
    image_cv2 = np.array(image_cv2)

    return image_cv2


def cv2_remove_white_background(cv2_image):
    _, cv2_image = cv2.threshold(cv2_image, 254, 255, cv2.THRESH_BINARY_INV)
    return cv2_image


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


def cv2_add_text_to_image(image, text, img_txt, background=False, padding=23):
    """
    Add text on cv2 images.

    :param image: cv2 image
    :param text: text string
    :param text_pos: xy start pos for text (same as cv2.putText(start_point))
    :param font_size: See cv2.putText(fontScale)
    :param bold: Write text in bold
    :param background: Add black background to text
    """

    text_dim, _ = cv2.getTextSize(text, img_txt.font, img_txt.size,
                                  img_txt.thickness)

    image, img_txt.pos = cv2_add_text_background(image, img_txt.pos, text_dim,
                                                 img_txt.size,
                                                 padding)
    # Add to image
    cv2.putText(image, text, img_txt.pos, img_txt.font, img_txt.size,
                img_txt.color_cv2, img_txt.thickness)
    return image


def cv2_add_text_background(image, pos, text_dim, font_size, padding=0,
                            color=(0, 0, 0)):
    """
    Adds text background box to cv2 images and adjusts text pos accordingly.

    :param image: cv2 image
    :param pos: xy start pos for text (same as cv2.putText(start_point))
    :param text_dim : w, h of text area (same as retval, _ from cv2.getTextSize())
    :param font_size: See cv2.putText(fontScale)
    :param padding: Adds padding to background box
    :param color: Color RGB values for box
    :return image: Image with box
    :return text_pos: Adjusted xy start pos for text (same as cv2.putText(start_point))
    """
    # Add background box for text
    box_end_pos = cv2_text_box_end_pos(pos, text_dim, padding)
    cv2.rectangle(image, pos, box_end_pos, color, -1)

    # Adjust text pos to middle of box
    box_x, box_y = pos
    _, text_h = text_dim
    padding_offset = int(padding / 2)
    text_pos = (box_x + padding_offset, box_y + text_h + font_size - 1 + padding_offset)

    return image, text_pos


def cv2_text_box_end_pos(pos, text_box, border=0):
    """
    Calculates end pos for a text box for cv2 images.

    :param pos: Position of text (same as for cv2 image)
    :param text_box: Size of text (same as for cv2 image)
    :param border: Outside padding of textbox
    :return box_end_pos: End xy coordinates for text box (end_point for cv2.rectangel())
    """
    box_x, box_y = pos
    text_w, text_h = text_box
    box_end_pos = (box_x + text_w + border, box_y + text_h + border)
    return box_end_pos


def cv2_gray_to_color(image):
    gray_as_color = cv2.merge([image, image, image])
    return gray_as_color


class Cv2ImageText():

    def __init__(self, font=cv2.FONT_HERSHEY_SIMPLEX, pos=(50, 50),
                 bold=False, background=False, size=1, padding=23, color=None):
        self.font = font
        self.size = size
        self.pos = pos
        self.bold = bold
        if bold:
            self.thickness = self.size * 3
        else:
            self.thickness = self.size * 2

        # Color
        if not color:
            color_info = sns.color_palette("gist_ncar_r")[1]
            self.color_matplotlib = color_info
            self.color_cv2 = [(1 - x) * 255 for x in color_info]


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
