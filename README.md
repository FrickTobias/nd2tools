[![CI Linux](https://github.com/FrickTobias/nd2tools/actions/workflows/ci_linux.yml/badge.svg)](https://github.com/FrickTobias/nd2tools/actions/workflows/ci_linux.yml) [![CI MacOS](https://github.com/FrickTobias/nd2tools/actions/workflows/ci_macos.yml/badge.svg)](https://github.com/FrickTobias/nd2tools/actions/workflows/ci_macos.yml)

# ND2-tools

A package for automating the most common ND2 exports, like to PNG images or MP4
timelapse videos. This is an ongoing development process where limitations and adding
features are continuously added. 

Feel free to post suggestions of what functionality to add by opening an issue.

### Current limitations

- Only handle one channel
- Scalebar lengths are not correct
- Only adds time information to images

## Contents

- [Prerequisites](#prerequisite)
- [Install](#install)
- [Usage](#usage)

## Prerequisite

Install miniconda according to the docs or use the script in the git repo.

```
bash install-miniconda.sh
```

## Install

#### 1. Clone the ND2-tools github

```
git clone https://github.com/FrickTobias/nd2tools.git 
```

#### 2. Install external package requirements into a virtual environment:

```
conda env create -n ndt -f environment.yml 
conda activate ndt
```

#### 3. Install the ND2-tools package:

```
pip install nd2tools 
```

#### Update by pulling from the git repository

```
git -C nd2tools pull
```

And then install the updated ND2-tools package again (see
step [3](#3-install-the-nd2-tools-package)).

Or install using editable (`pip install -e nd2tools`) mode and you only have to use the
`git -C nd2tools pull` command

## Usage

See `nd2tools -h` and `nd2tools [display|image|movie] -h` .

## Examples

#### Make timelapse

```
nd2tools movie cells.nd2 timelapse.mp4
```

#### Write image

```
nd2tools image cells.nd2 image.png
```

#### View image

```
nd2tools display cells.nd2 
```

#### Split images into a 4x4 grid

```
nd2tools [display|image|movie] cells.nd2 output --split 4 4 --keep 0 0
```