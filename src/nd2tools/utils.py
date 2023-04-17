import numpy as np
import numpy.typing as npt

def convert_bit_depths(image: npt.NDArray, bit_depths: str, min_val: int = None,
                       max_val: int = None) -> npt.NDArray:
    # Set conversion factor values
    if bit_depths == "8bit":
        bit_max = 255
        np_bit_depths = np.uint8
    elif bit_depths == "16bit":
        bit_max = 16383
        np_bit_depths = np.uint16

    # Calculate max/min value for image
    if not max_val:
        max_val = np.max(image)
    if not min_val:
        min_val = np.min(image)

    # Convert image
    image_converted = ((image - min_val) / (max_val - min_val) * bit_max).astype(
        np_bit_depths)

    return image_converted
