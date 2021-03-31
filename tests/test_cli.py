from pathlib import Path
import pytest
import shutil
import subprocess
import sys

from nd2tools.cli import scalebar

TESTDATA = Path("tests/testdata")
TESTDATA_IMAGE = TESTDATA / "img.bmp"


def test_environment():
    tools = [
        "python --version",
    ]
    for tool in tools:
        print(f"'$ {tool}'")
        subprocess.run(tool.split(" "), stderr=sys.stdout)


def test_scalebar(workdir):
    scalebar(TESTDATA_IMAGE)
