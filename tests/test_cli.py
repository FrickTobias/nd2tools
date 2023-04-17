from pathlib import Path
import subprocess
import sys
import hashlib
from nd2tools.cli.image import image as ndt_image
from nd2tools.cli.movie import movie as ndt_movie

# Test data
TESTDATA = Path("tests/test-data")
ND2_TEST_IMAGE = TESTDATA / "cells.nd2"

# Input args
INPUT_IMAGE = ND2_TEST_IMAGE
INPUT_MOVIE = ND2_TEST_IMAGE

# Output args
OUTPUT_IMAGE = TESTDATA / "cells-images"
OUTPUT_MOVIE = TESTDATA / "cells.mp4"

# Expected output paths
EXPECTED_OUT_IMAGE = OUTPUT_IMAGE / "image.tif"
EXPECTED_OUT_MOVIE = OUTPUT_MOVIE

# md5sums
MD5SUM_MOVIE = "90022ed5f2c7752af56b8a2b6eebe01c"
MD5SUM_IMAGE = "5419cdc2d6f85495a4b56869c584e4a8"
#MD5SUM_SCALEBAR = "f21298ea43f89c2c5bea87bf32b6ef62"
#MD5SUM_TIMESTAMPS = "3ac609026db06f3cb00148042e6200eb"


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


def test_image(nd2=INPUT_IMAGE, outdir=OUTPUT_IMAGE, outfile=EXPECTED_OUT_IMAGE, md5sum=MD5SUM_IMAGE):
    ndt_image(input_file=nd2, output_folder=outdir)
    assert md5sum == get_md5sum(outfile)
    return


def test_movie(nd2=INPUT_MOVIE, out=OUTPUT_MOVIE, outfile=EXPECTED_OUT_MOVIE, md5sum=MD5SUM_MOVIE):
    ndt_movie(input=nd2, output=out)
    assert md5sum == get_md5sum(outfile)
    return

#def test_scalebar(nd2_image=TESTDATA_IMAGE, img_out=TEST_OUT_IMAGE,
#                  md5sum=MD5SUM_SCALEBAR):
#    ndt_image(input_file=nd2_image, output_folder=img_out)#, scalebar=True)
#    #assert md5sum == get_md5sum(img_out)
#    return
#
#
#def test_timestamps(nd2_image=TESTDATA_IMAGE, img_out=TEST_OUT_IMAGE,
#                    md5sum=MD5SUM_TIMESTAMPS):
#    ndt_image(input_file=nd2_image, output_folder=img_out)#, timestamps=True)
#    #assert md5sum == get_md5sum(img_out)
#    return
#