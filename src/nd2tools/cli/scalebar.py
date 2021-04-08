"""
Adds scalebars to images
"""

import matplotlib.pyplot as plt
import imageio
import pathlib
import logging


from matplotlib_scalebar.scalebar import ScaleBar

logger = logging.getLogger(__name__)


def main(args):
    # Get screen pixel density
    dpi = get_screen_dpi()

    # Load image and extract
    im = imageio.imread(args.image)
    width, height, channels = im.shape

    # Compensate size of pixel for objective magnification
    pixel_size_real = args.pixelsize / args.magnification

    # Create subplot
    # Specifies figsize and dpi in order to keep original resolution
    fig, ax = plt.subplots(figsize=(height / dpi, width / dpi), dpi=dpi)
    ax.axis("off")

    # Plot image
    ax.imshow(im, cmap="gray")

    # Scale bar settings
    if args.big:
        scalebar_width = 0.015
        scalebar_length = 0.3
        font = {
            "size": "40"
        }
    else:
        scalebar_width = 0.01
        scalebar_length = 0.2
        font = {}

    # Create scale barnan
    scalebar = ScaleBar(pixel_size_real, "um", frameon=False,
                        length_fraction=scalebar_length,
                        width_fraction=scalebar_width, font_properties=font)
    ax.add_artist(scalebar)

    # Save to out
    plt.savefig(args.output, format="jpeg", bbox_inches='tight', pad_inches=0)


def get_screen_dpi():
    # Get dpi
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    screen = app.screens()[0]
    dpi = screen.physicalDotsPerInch()
    app.quit()
    return dpi


def add_arguments(parser):
    parser.add_argument(
        "image", type=pathlib.Path,
        help="Input image to add scale bar to."
    )
    parser.add_argument(
        "pixelsize", type=float,
        help="Pixel size in sensor in um."
    )
    parser.add_argument(
        "magnification", type=float,
        help="Magnification from objective."
    )
    parser.add_argument(
        "--big", "-b", action="store_true",
        help="Creates larger scale bar."
    )
    parser.add_argument(
        "output",
        help="Output file name. Will save in jpeg."
    )
