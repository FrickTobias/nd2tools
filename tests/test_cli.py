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
TEST_OUT_PREFIX = TESTDATA / "test-output"
DISPLAY_DURATION_MS = 1


def test_environment():
    tools = [
        "python --version",
    ]
    for tool in tools:
        print(f"'$ {tool}'")
        subprocess.run(tool.split(" "), stderr=sys.stdout)


def test_image(nd2_image=TESTDATA_IMAGE, img_out=TEST_OUT_PREFIX):
    ndt_image(input=nd2_image, output=img_out)
    # Add test
    return


def test_movie(nd2_image=TESTDATA_IMAGE, movie_out=TEST_OUT_PREFIX):
    ndt_movie(input=nd2_image, output=movie_out)
    # Add test
    return


def test_display(nd2_image=TESTDATA_IMAGE, duration=DISPLAY_DURATION_MS):
    ndt_display(input=nd2_image, duration=duration)
    # ADD test
    return
