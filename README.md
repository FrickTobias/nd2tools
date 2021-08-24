[![CI Linux](https://github.com/FrickTobias/nd2tools/actions/workflows/ci_linux.yaml/badge.svg?branch=main&event=schedule)](https://github.com/FrickTobias/nd2tools/actions/workflows/ci_linux.yaml) [![CI MacOS](https://github.com/FrickTobias/nd2tools/actions/workflows/ci_macos.yaml/badge.svg?branch=main&event=schedule)](https://github.com/FrickTobias/nd2tools/actions/workflows/ci_macos.yaml)

# ND2-tools

ND2 image handling

- [Install](#install)
- [Usage](#usage)

## Install

1. Clone the ND2-tools github
    ```
    git clone https://github.com/FrickTobias/nd2tools.git 
    ```

2. Install the base environment for `environment.yml` file using conda:
    ```
    conda env create -n nd2tools -f environment.yml 
    ```

3. Install the ND2-tools package:
    ```
    pip install . 
    ```
   For development the package can be installed in editable mode
   using `pip install -e .`.
   
## Usage 

See `nd2tools -h` or `nd2tools [display|image|movie] -h` .

## Examples

Make timelapse
```
nd2tools movie cells.nd2 timelapse.mp4
```

Convert to normal image file
```
nd2tools image cells.nd2 image.png
```

View an image
```
nd2tools display cells.nd2 
```

Cropping images
```
nd2tools [display|image|movie] cells.nd2 output --split 4 4 --keep 0 0
```