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

### Install

1. Clone the ND2-tools github 
    ```
    git clone https://github.com/FrickTobias/nd2tools.git
    ```

2. Install dependencies
    ```
    conda env create -n ndt -f environment.yml
    conda activate ndt
    ```

3. Install the ND2-tools package:
    ```
    pip install nd2tools
    ```
   
## Usage 

See `nd2tools -h`.
