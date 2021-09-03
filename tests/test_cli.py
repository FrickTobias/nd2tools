from pathlib import Path
# import pytest
# import shutil
import subprocess
import sys
from nd2tools.cli.display import display as ndt_display
from nd2tools.cli.image import image as ndt_image
from nd2tools.cli.movie import movie as ndt_movie

# from nd2tools.cli import scalebar

TESTDATA = Path("test-data")
TESTDATA_IMAGE = TESTDATA / "z-axis.nd2"
TEST_OUT_IMAGE = TESTDATA / "z-axis.png"
TEST_OUT_MOVIE = TESTDATA / "z-axis.mp4"
IMAGE_OUT_NAME = TESTDATA / "z-axis.image-6.png"
DISPLAY_DURATION_MS = 1


def test_environment():
    tools = [
        "python --version",
    ]
    for tool in tools:
        print(f"'$ {tool}'")
        subprocess.run(tool.split(" "), stderr=sys.stdout)


def test_display(nd2_image=TESTDATA_IMAGE, duration=DISPLAY_DURATION_MS):
    ndt_display(input=nd2_image, duration=duration)
    # ADD test
    return


def test_image(nd2_image=TESTDATA_IMAGE, img_out=TEST_OUT_IMAGE):
    ndt_image(input=nd2_image, output=img_out, clip_start=5, clip_end=5)
    assert IMAGE_OUT_NAME.exists()
    return


def test_movie(nd2_image=TESTDATA_IMAGE, movie_out=TEST_OUT_MOVIE):
    ndt_movie(input=nd2_image, output=movie_out, clip_start=5, clip_end=5)
    assert movie_out.exists()
    return


def test_scalebar(nd2_image=TESTDATA_IMAGE, img_out=TEST_OUT_IMAGE):
    ndt_image(input=nd2_image, output=img_out, clip_start=5, clip_end=5, scalebar=True)
    assert IMAGE_OUT_NAME.exists()
    return


def test_timestamps(nd2_image=TESTDATA_IMAGE, img_out=TEST_OUT_IMAGE):
    ndt_image(input=nd2_image, output=img_out, clip_start=5, clip_end=5,
              timestamps=True)
    assert IMAGE_OUT_NAME.exists()
    return
