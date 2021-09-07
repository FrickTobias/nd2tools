from pathlib import Path
# import pytest
# import shutil
import subprocess
import sys
import hashlib
from nd2tools.cli.display import display as ndt_display
from nd2tools.cli.image import image as ndt_image
from nd2tools.cli.movie import movie as ndt_movie

# from nd2tools.cli import scalebar

TESTDATA = Path("test-data")
TESTDATA_IMAGE = TESTDATA / "z-axis.nd2"

TEST_OUT_IMAGE = TESTDATA / "z-axis.png"
TEST_OUT_MOVIE = TESTDATA / "z-axis.mp4"
IMAGE_OUT_NAME = TESTDATA / "z-axis.image-6.png"

MD5SUM_IMAGE = "3b4267f85f9adf0626139490b63e753b"
MD5SUM_MOVIE = "906282884ce6d4944256293898846af9"
MD5SUM_SCALEBAR = "eb1cae2e75f461922f5e45b6baed2b85"
MD5SUM_TIMESTAMPS = "092ba7536dad27f4e0146b3ac9edf8fd"

DISPLAY_DURATION_MS = 1

def get_md5sum(file):
    md5_hash = hashlib.md5()
    with open(file, "rb") as openin:
        for line in openin:
            md5_hash.update(line)
        digest = md5_hash.hexdigest()
    return digest


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


def test_image(nd2_image=TESTDATA_IMAGE, img_out=TEST_OUT_IMAGE, md5sum=MD5SUM_IMAGE):
    ndt_image(input=nd2_image, output=img_out, clip_start=5, clip_end=5)
    assert md5sum == get_md5sum(IMAGE_OUT_NAME)
    return


def test_movie(nd2_image=TESTDATA_IMAGE, movie_out=TEST_OUT_MOVIE, md5sum=MD5SUM_MOVIE):
    ndt_movie(input=nd2_image, output=movie_out, clip_start=5, clip_end=5)
    assert md5sum == get_md5sum(movie_out)
    return


def test_scalebar(nd2_image=TESTDATA_IMAGE, img_out=TEST_OUT_IMAGE, md5sum=MD5SUM_SCALEBAR):
    ndt_image(input=nd2_image, output=img_out, clip_start=5, clip_end=5, scalebar=True)
    assert md5sum == get_md5sum(IMAGE_OUT_NAME)
    return


def test_timestamps(nd2_image=TESTDATA_IMAGE, img_out=TEST_OUT_IMAGE, md5sum=MD5SUM_TIMESTAMPS):
    ndt_image(input=nd2_image, output=img_out, clip_start=5, clip_end=5,
              timestamps=True)
    assert md5sum == get_md5sum(IMAGE_OUT_NAME)
    return
