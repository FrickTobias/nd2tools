[![CI Linux](https://github.com/FrickTobias/nd2tools/actions/workflows/ci_linux.yml/badge.svg)](https://github.com/FrickTobias/nd2tools/actions/workflows/ci_linux.yml) [![CI MacOS](https://github.com/FrickTobias/nd2tools/actions/workflows/ci_macos.yml/badge.svg)](https://github.com/FrickTobias/nd2tools/actions/workflows/ci_macos.yml)


![nd2tools-logo](images/nd2tools-logo.png?raw)

A package for automating the most common ND2 exports, like to PNG images or MP4
timelapse videos. This is an ongoing development process where limitations and adding
features are continuously added.

![nd2tools-workflow](images/nd2tools-workflow.png)

Feel free to post suggestions of what functionality to add by opening an issue.

## Contents

- [Prerequisites](#prerequisite)
- [Install](#install)
- [Usage](#usage)

## Requirements

Install
miniconda [according to their instructions](https://docs.conda.io/en/latest/miniconda.html)
or use the script in this git repo.

```
bash install-miniconda.sh
```

## Setup

#### 1. Clone the ND2-tools github

```
git clone https://github.com/FrickTobias/nd2tools.git 
```

#### 2. Install external package requirements into a virtual environment

```
conda env create -n ndt -f nd2tools/environment.yml 
conda activate ndt
```

#### 3. Install the ND2-tools package

```
pip install -e nd2tools 
```

#### 4. Test installation

Install `pytest` and run the testing scripts in `tests/.

```
conda install -c anaconda pytest
pytest nd2tools/tests
```

#### Update by pulling from the git repository

```
git -C nd2tools pull
```

## Usage

See `nd2tools -h` and `nd2tools [display|image|movie] -h` .

## Examples

#### View image

```
nd2tools display cells.nd2 
```

#### Write image

```
nd2tools image cells.nd2 cells.png
```

#### Make timelapse

```
nd2tools movie cells.nd2 cells.mp4
```

#### Split images into a 4x4 grid

```
nd2tools [image|movie] cells.nd2 cells.mp4 --split 4 4 --keep 0 0
```

#### Add scalebar and timestamps

```
nd2tools [display|image|movie] cells.nd2 cells.mp4 --scalebar --timestamps
```
