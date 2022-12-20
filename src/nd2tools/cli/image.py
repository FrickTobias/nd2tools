"""
Writes images from nd2 files
"""

import pathlib
import logging
import cv2
from tqdm import tqdm
from nd2reader import ND2Reader

from nd2tools.utils import ImageCoordinates
from nd2tools.utils import map_uint16_to_uint8
from nd2tools.utils import generate_filename
from nd2tools.utils import ScalingMinMax
from nd2tools.utils import add_global_args
from nd2tools.utils import add_clipping_options

from nd2tools import cv2_utils

from nd2tools.utils import nd2_get_time

logger = logging.getLogger(__name__)


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
        "--format", type=str, default="tif",
        help="Output format. Will be appended to output name if not included. Default: "
             "%(default)s."
    )
    # TODO: Move to utils and standardize for all moduels of nd2tools
    parser.add_argument("-z", "--z-level", type=int,
                        help="Z level. Change z level for image output. Default: "
                             "%(default)s.")
    parser.add_argument("-t", "--timepoint", type=int,
                        help="Timepoint. Only extract image number -t.")
    parser.add_argument("--iter-axes", type=str,
                        help="Manually set iter axes. Possible values depend on nd2 "
                             "image and will be printed when running script.")
    #parser.add_argument("--iter-axes", type=str, default="t",
    #                    help="Define which axes to iterate over. Default: %(default)s.")


def main(args):
    image(input=args.input, output=args.output, format=args.format,
          clip_start=args.clip_start, clip_end=args.clip_end, split=args.split,
          keep=args.keep, cut=args.cut, trim=args.trim,
          scalebar_length=args.scalebar_length, timestamps=args.timestamps,
          scalebar=args.scalebar, z_level=args.z_level, timepoint=args.timepoint,
          iter_axes=args.iter_axes)


def image(input, output, format="tif", clip_start=0, clip_end=0, split=None,
          keep=None, cut=None, trim=None, scalebar=None, scalebar_length=None,
          timestamps=None, z_level=0, timepoint=None, iter_axes=None):
    with ND2Reader(input) as images:
        logger.info(f"Image info: {images}")
        img_txt = cv2_utils.ImageText()
        timesteps = nd2_get_time(images)
        im_xy = ImageCoordinates(x1=0, x2=images.sizes['x'], y1=0,
                                 y2=images.sizes['y'])
        im_xy.adjust_frame(split, keep, cut, trim)
        frame_pos_list = im_xy.frames()
        scaling_min_max = ScalingMinMax(mode="continuous",
                                        image=images[0])
        pixel_size = images[0].metadata["pixel_microns"]
        first_frame = clip_start
        last_frame = len(images) - clip_end
        assert images[0].dtype == "uint16", f"Not 16bit image ({images[0].dtype})"

        # TODO: Implement this properly (iter axis choice etc)
        if z_level:
            images.default_coords["z"] = z_level
        images.default_coords["t"] = timepoint

        possible_iter_axes = "".join(set(images.axes) - set(["x", "y"]))
        logger.info(f"Possible iter axes for current file: {possible_iter_axes}")
        if iter_axes:
            logger.info(f"Manually setting iter axes")
            images.iter_axes = iter_axes
        logger.info(f"Iter axes: {images.iter_axes}")

        for image_number, image in enumerate(tqdm(images[first_frame:last_frame],
                                                  desc="Writing image file(s)",
                                                  unit=" images",
                                                  total=last_frame - first_frame),
                                             start=first_frame):

            #            import pdb
            #            pdb.set_trace()

            if scaling_min_max.mode == "continuous" or \
                    scaling_min_max.mode == "current":
                scaling_min_max.update(image)
            image = map_uint16_to_uint8(image,
                                        lower_bound=scaling_min_max.min_current,
                                        upper_bound=scaling_min_max.max_current)
            if timestamps:
                acquisition_time = timesteps[image_number]

            # If splitting image, iterate over frames
            for frame_fraction, frame_pos in enumerate(frame_pos_list):
                image_crop = cv2_utils.crop_image(image, frame_pos)
                image_crop = cv2_utils.gray_to_color(image_crop)

                if scalebar:
                    image_crop = cv2_utils.add_scalebar(image_crop, pixel_size,
                                                        color=img_txt.color_cv2,
                                                        length=scalebar_length)
                if timestamps:
                    image_crop = cv2_utils.add_text_to_image(image_crop,
                                                             f"t: {acquisition_time}",
                                                             pos=img_txt.pos,
                                                             color=img_txt.color_cv2,
                                                             background=True)

                # Generate filename and write to out
                metadata = build_metadata_string(images, image_number,
                                                 frame_pos_list,
                                                 frame_fraction)
                file_path = generate_filename(output, metadata=metadata,
                                              format=format)
                cv2.imwrite(file_path, image_crop)


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
