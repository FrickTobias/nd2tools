import numpy as np
import logging
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

IMAGE_SEPARATOR = 10

logger = logging.getLogger(__name__)


def image_pos_generator(string_description, height, width, padding=(50, 50)):
    y_string, x_string = string_description.split()
    x_padding, y_padding = padding

    if x_string == "left":
        x_pos = x_padding
        reverse = False
    elif x_string == "right":
        x_pos = width - x_padding
        reverse = True
    else:
        raise ValueError("x_string not 'left' or 'right'")

    if y_string == "top":
        y_pos = y_padding
    elif y_string == "bottom":
        y_pos = height - y_padding
    else:
        raise ValueError("y_string not 'top' or 'bottom'")

    return (x_pos, y_pos), reverse


def scalebar_end_pos(pos, px_size, im_height, im_width, height_frac=0.01,
                     width_frac=0.1, length=None, decimals=2, reverse=False):
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

    # Compensate for direction (if put on right side)
    if reverse:
        delta_x = - delta_x

    # Convert to px positions in image
    x, y = pos
    end_pos = (x + delta_x, y + delta_y)

    return end_pos, x_size


def gray_to_color(image):
    gray_as_color = cv2.merge([image, image, image])
    return gray_as_color


def remove_white_background(cv2_image):
    _, cv2_image = cv2.threshold(cv2_image, 254, 255, cv2.THRESH_BINARY_INV)
    return cv2_image


def add_text_to_image(image, text, font=cv2.FONT_HERSHEY_SIMPLEX, size=1,
                      pos=(50, 50),
                      color=(0, 0, 0), background=None, padding=23):
    """
    Add text on cv2 images.

    :param image: cv2 image
    :param text: text string
    :param text_pos: xy start pos for text (same as cv2.putText(start_point))
    :param font_size: See cv2.putText(fontScale)
    :param bold: Write text in bold
    :param background: Add black background to text
    """

    # Add box
    thickness = size * 2
    if background:
        text_dim, _ = cv2.getTextSize(text, font, size, thickness)
        image, pos = add_text_background(image, pos, text_dim, size,
                                         padding)

    # Add text
    cv2.putText(image, text, pos, font, size, color, thickness)

    return image


def add_text_background(image, pos, text_dim, font_size, padding=0,
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
    # Add box to image
    box_end_pos = text_box_end_pos(pos, text_dim, padding)
    cv2.rectangle(image, pos, box_end_pos, color, -1)

    # Calculate text pos adjusted to middle of box
    box_x, box_y = pos
    _, text_h = text_dim
    padding_offset = int(padding / 2)
    text_pos = (box_x + padding_offset, box_y + text_h + font_size - 1 + padding_offset)

    return image, text_pos


def text_box_end_pos(pos, text_box, border=0):
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


def crop_image(image, frame_pos):
    x1, x2, y1, y2 = frame_pos
    image_crop = image[y1:y2, x1:x2]
    return image_crop


def add_scalebar(image, px_size, position="top right", color=(0, 0, 0),
                 length=None):
    """
    """

    # Get variables
    img_txt = ImageText()
    im_height, im_width, _ = image.shape
    pos, reverse = image_pos_generator(position, im_height, im_width)

    # Add box to image
    box_end_pos, real_length = scalebar_end_pos(pos, px_size=px_size,
                                                im_height=im_height,
                                                im_width=im_width, length=length,
                                                reverse=reverse)

    # Switches start/stop if position is "_ right"
    if reverse:
        pos, box_end_pos = _switch_x_in_tuple(pos, box_end_pos)

    cv2.rectangle(image, pos, box_end_pos, img_txt.color_cv2, -1)

    # String under box
    text = f"{real_length} um"

    # Calculate text position
    box_x1, box_y1 = pos
    box_x2, box_y2 = box_end_pos
    text_dim, _ = cv2.getTextSize(text, img_txt.font, img_txt.size, img_txt.thickness)
    _, text_h = text_dim
    txt_pos = (box_x1, box_y2 + text_h + img_txt.size - 1 + IMAGE_SEPARATOR)

    # Add text to image
    image = add_text_to_image(image, text, size=img_txt.size, font=img_txt.font,
                              color=img_txt.color_cv2, pos=txt_pos)
    return image


def _switch_x_in_tuple(tuple1, tuple2):
    x1, y1 = tuple1
    x2, y2 = tuple2
    tuple1 = (x2, y1)
    tuple2 = (x1, y2)
    return tuple1, tuple2


class ImageText():

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
