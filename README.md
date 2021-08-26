[![CI Linux](https://github.com/FrickTobias/nd2tools/actions/workflows/ci_linux.yaml/badge.svg?branch=main&event=schedule)](https://github.com/FrickTobias/nd2tools/actions/workflows/ci_linux.yaml) 
[![CI MacOS](https://github.com/FrickTobias/nd2tools/actions/workflows/ci_macos.yaml/badge.svg?branch=main&event=schedule)](https://github.com/FrickTobias/nd2tools/actions/workflows/ci_macos.yaml)

:exclamation:**NB! This is currently under heavy development.**:exclamation:

# ND2-tools

ND2 image handling

- [Install](#setup)
- [Usage](#usage)

## Setup

### Prerequisites

- [miniconda](https://docs.conda.io/en/latest/miniconda.html)

```
if [[ $OSTYPE = "linux-gnu" ]]; then
    wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
elif [[ $OSTYPE = "darwin"* ]]; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh
fi
bash miniconda.sh -b -p $HOME/miniconda
source $HOME/miniconda/etc/profile.d/conda.sh
```

### Install

#### 1. Clone the ND2-tools github
```
git clone https://github.com/FrickTobias/nd2tools.git 
```

#### 2. Install external package requirements into a virtual environment:
```
conda env create -n nd2tools -f environment.yml 
```

#### 3. Install the ND2-tools package:
```
pip install nd2tools 
```
   
#### Update by pulling from the git repository
```
git -C nd2tools pull
```

And then install the updated ND2-tools package again (see step [3](#3-install-the-nd2-tools-package)). 

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
