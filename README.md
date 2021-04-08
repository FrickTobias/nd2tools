[![CI](https://github.com/FrickTobias/nd2tools/workflows/CI/badge.svg?branch=main)](https://github.com/FrickTobias/nd2tools/actions?query=branch%3Amain)

# ND2-tools

ND2 image handling

- [Install](#install)
- [Recipies](#Recipies)

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

## Recipies

### Adding scalebar to image

nd2tools scalebar <input-img.bmp> -bt -o <output-img.png>
