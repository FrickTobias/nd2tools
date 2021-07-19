from pathlib import Path
# import pytest
# import shutil
import subprocess
import sys

# from nd2tools.cli import scalebar

TESTDATA = Path("tests/testdata")
TESTDATA_IMAGE = TESTDATA / "img.bmp"
TESTOUT_SCALEBAR = TESTDATA / "img-scalebar.jpeg"


def test_environment():
    tools = [
        "python --version",
    ]
    for tool in tools:
        print(f"'$ {tool}'")
        subprocess.run(tool.split(" "), stderr=sys.stdout)


def test_scalebar():
    return

def test_crop():
    return

def test_split():
    return

    # scalebar.scalebar(TESTDATA_IMAGE, 3.45, 10, TESTOUT_SCALEBAR)
