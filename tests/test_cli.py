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

TESTDATA = Path("tests/test-data")
TESTDATA_IMAGE = TESTDATA / "cells.nd2"
TEST_OUT_IMAGE = TESTDATA / "cells.tif"
TEST_OUT_MOVIE = TESTDATA / "cells.mp4"

MD5SUM_IMAGE = "097aa78e25a161a5fbd3a57e646505e0"
MD5SUM_MOVIE = "90022ed5f2c7752af56b8a2b6eebe01c"
MD5SUM_SCALEBAR = "f21298ea43f89c2c5bea87bf32b6ef62"
MD5SUM_TIMESTAMPS = "3ac609026db06f3cb00148042e6200eb"

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
    ndt_display(input=nd2_image, duration=duration, display=False)
    # ADD test
    return


def test_image(nd2_image=TESTDATA_IMAGE, img_out=TEST_OUT_IMAGE, md5sum=MD5SUM_IMAGE):
    ndt_image(input=nd2_image, output=img_out)
    assert md5sum == get_md5sum(img_out)
    return


def test_movie(nd2_image=TESTDATA_IMAGE, movie_out=TEST_OUT_MOVIE, md5sum=MD5SUM_MOVIE):
    ndt_movie(input=nd2_image, output=movie_out)
    assert md5sum == get_md5sum(movie_out)
    return


def test_scalebar(nd2_image=TESTDATA_IMAGE, img_out=TEST_OUT_IMAGE,
                  md5sum=MD5SUM_SCALEBAR):
    ndt_image(input=nd2_image, output=img_out, scalebar=True)
    assert md5sum == get_md5sum(img_out)
    return


def test_timestamps(nd2_image=TESTDATA_IMAGE, img_out=TEST_OUT_IMAGE,
                    md5sum=MD5SUM_TIMESTAMPS):
    ndt_image(input=nd2_image, output=img_out, timestamps=True)
    assert md5sum == get_md5sum(img_out)
    return
