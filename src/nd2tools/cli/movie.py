"""
Converts images to MP4 movies
"""

import pathlib
import logging
import time
import cv2
import numpy as np
from tqdm import tqdm
import seaborn as sns

from nd2reader import ND2Reader

from nd2tools.utils import ImageCoordinates
from nd2tools.utils import map_uint16_to_uint8
from nd2tools.utils import generate_filename

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
    open_video_files = OpenVideoFiles(file_prefix, fps, width, height, frame_pos_list,
                                      is_color=1)

    # Writing outputs
    scaling_min_max = ScalingMinMax(mode=conversion_method, scaling=scale_conversion,
                                    images=images)
    first_frame = clip_start
    last_frame = len(images) - clip_stop
    timesteps = get_time(images)
    # timesteps = [datetime(timestep) for timestep in timesteps]
    for frame_number, image in enumerate(
            tqdm(images, desc=f"Writing movie file(s)", unit=" images",
                 total=last_frame)):

        # Option: --clip-start
        if frame_number < first_frame:
            continue

        # Split image and writes to appropriate files
        acquisition_time = timesteps[frame_number]
        for frame_pos in frame_pos_list:

            # Crop image
            x1, x2, y1, y2 = frame_pos
            image = image[y1:y2, x1:x2]

            # convert 16bit to 8bit
            if image.dtype == "uint16":
                if scaling_min_max.mode == "continuous" or scaling_min_max.mode == "current":
                    logger.info(f"frame: {frame_number}")
                    scaling_min_max.update(image_cropped)
                image = map_uint16_to_uint8(image,
                                            lower_bound=scaling_min_max.min_current,
                                            upper_bound=scaling_min_max.max_current)

            # Add text to frames
            image = gray_to_color(image)  # Convert to color (copy to 3 channels)
            image = add_text_to_image(image, text=f"t: {acquisition_time}",
                                      background=True)

            # TODO: Add scalebar functionality
            #image = add_scalebar(image, pixel_size=2, magnification=10)

            # Write image
            open_video_files.dictionary[frame_pos].write(image)

        # Option: --clip-end
        if frame_number >= last_frame:
            break

    # Close files
    open_video_files.close()




## TODO: Get this funcional
#def add_scalebar(image, pixel_size, magnification):
#    #import matplotlib.pyplot as plt
#    #import matplotlib.cbook as cbook
#    #from matplotlib_scalebar.scalebar import ScaleBar
#    #plt.figure()
#    #image = plt.imread(cbook.get_sample_data('grace_hopper.png'))
#    #f = plt.imshow(image)
#    #scalebar = ScaleBar(0.2)  # 1 pixel = 0.2 meter
#    #plt.gca().add_artist(scalebar)
#    #plt.gca().add_artist(scalebar)
#    #import pdb
#    #pdb.set_trace()
#    #return f
#
#
#    #import numpy as np
#    #col_1 = np.vstack([image, image])
#    #cv2.imshow("test", col_1)
#    #cv2.waitKey(0)
#
#    from matplotlib_scalebar.scalebar import ScaleBar
#    import matplotlib as plt
#    scalebar_width = 0.01
#    scalebar_length = 0.2
#    font = {}
#
#    # Get screen pixel density
#    dpi = get_screen_dpi()
#
#    # Load image and extract
#    width, height, channels = image.shape
#
#    # Compensate size of pixel for objective magnification
#    pixel_size_real = pixel_size / magnification
#
#    # Create subplot
#    # Specifies figsize and dpi in order to keep original resolution
#    fig, ax = plt.pyplot.subplots(figsize=(height / dpi, width / dpi), dpi=dpi)
#    ax.axis("off")
#
#    # Plot image
#    ax.imshow(image, cmap="gray")
#
#    # Create scale barnan
#    scalebar = ScaleBar(pixel_size_real, "um", frameon=False,
#                        length_fraction=scalebar_length,
#                        width_fraction=scalebar_width, font_properties=font)
#    ax.add_artist(scalebar)
#    #import pdb
#    #pdb.set_trace()
#
#    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
#    from matplotlib.figure import Figure
#
#    canvas = FigureCanvas(fig)
#
#    canvas.draw()  # draw the canvas, cache the renderer
#    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
#
#    import pdb
#    pdb.set_trace()
#    #plt.pyplot.show()
#
#
#
#    return image
#    # Save to out
#    # plt.pyplot.savefig(output, format="jpeg", bbox_inches='tight', pad_inches=0)
#
#
## TODO: Get this funcional
#def get_screen_dpi():
#    # Get dpi
#    import sys
#    from PyQt5.QtWidgets import QApplication
#    app = QApplication(sys.argv)
#    screen = app.screens()[0]
#    dpi = screen.physicalDotsPerInch()
#    app.quit()
#    return dpi


def get_time(images, format_string="%H:%M:%S"):
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


# TODO: Do this properly
def add_text_to_image(image, text, text_pos=(50, 50), font_size=1, bold=False,
                      box=False, background=False):
    """
    Adds yellow text to top left of image
    """

    # Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    if bold:
        thickness = font_size * 3
    else:
        thickness = font_size * 2

    # Color
    color_info = sns.color_palette("gist_ncar_r")[1]
    text_color = [(1 - x) * 255 for x in color_info]

    # Add black box
    if background:
        text_size, _ = cv2.getTextSize(text, font, font_size, thickness)
        image, text_pos = add_text_background(image, text_pos, text_size, font_size,
                                              border=23)

    # Add to image
    cv2.putText(image, text, text_pos, font, font_size, text_color, thickness)
    return image


# TODO: Do this properly
def add_text_background(image, pos, text_size, font_size, border=0, color=(0, 0, 0)):
    # Add box to image
    box_x, box_y = pos
    text_w, text_h = text_size
    box_dimensions = (box_x + text_w + border, box_y + text_h + border)
    cv2.rectangle(image, pos, box_dimensions, color, -1)

    # Adjust text pos to middle of box
    border_offset = int(border / 2)
    text_pos = (box_x + border_offset, box_y + text_h + font_size - 1 + border_offset)

    return image, text_pos


def gray_to_color(image):
    gray_as_color = cv2.merge([image, image, image])
    return gray_as_color


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
