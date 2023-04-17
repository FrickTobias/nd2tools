import sys
import hashlib
import subprocess
from pathlib import Path

from nd2tools.cli.image import image as ndt_image
from nd2tools.cli.movie import movie as ndt_movie

# Test data
TESTDATA = Path("tests/test-data")
ND2_TEST_IMAGE = TESTDATA / "cells.nd2"

# Input args
INPUT_IMAGE = ND2_TEST_IMAGE
INPUT_MOVIE = ND2_TEST_IMAGE

# Output args
OUTPUT_IMAGE = TESTDATA / "images"
OUTPUT_MOVIE = TESTDATA / "movies"

# Expected output paths
EXPECTED_OUT_IMAGE = OUTPUT_IMAGE / "image.tif"
EXPECTED_OUT_MOVIE = OUTPUT_MOVIE / "movie.mp4"

# md5sums
MD5SUM_MOVIE = "c5ac318b1799ce79e7ba7ea59b6effaa"
MD5SUM_IMAGE = "5419cdc2d6f85495a4b56869c584e4a8"


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


def test_image(nd2=INPUT_IMAGE, outdir=OUTPUT_IMAGE, outfile=EXPECTED_OUT_IMAGE,
               md5sum=MD5SUM_IMAGE):
    ndt_image(input_file=nd2, output_folder=outdir)
    assert md5sum == get_md5sum(outfile)
    return


def test_movie(nd2=INPUT_MOVIE, out=OUTPUT_MOVIE, outfile=EXPECTED_OUT_MOVIE,
               md5sum=MD5SUM_MOVIE):
    ndt_movie(input_file=nd2, output_folder=out)
    assert md5sum == get_md5sum(outfile)
    return
