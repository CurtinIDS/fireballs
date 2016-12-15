# Fireballs

Data processing and analysis scripts for detecting fireballs from optical and radio astronomy images for the Curtin Institute for Radio Astronomy (CIRA).

**cache/**

Preprocessed datasets generated from raw data

**data/**

Raw data files. Datasets that are sensitive or large are not committed to this remote repository. Data must be stored locally in this folder to run the scripts.

**docs/**

Project related documents

**output/**

Output files generated from scripts

**results/**

Machine learning experiment results

**src/**

Data processing and statistical analysis scripts

## Contributors:
* Curtin Institute for Computation (CIC)
 * Kevin Chai (k.chai@curtin.edu.au)
 * Shiv Meka (shiv.meka@curtin.edu.au)

* CIRA
 * Paul Hancock (paul.hancock@curtin.edu.au)
 * Xiang Zhang (xiang.zhang11@postgrad.curtin.edu.au)

* Desert Fireball Network (DFN)
 * Hadrien Devillepoix (hadriendvpx@gmail.com)
 

## Installation
`pip install -r requirements.txt`

## Usage

### Data pre-processsing
The code works on JPEG images. Images should be copied to a folder within the `cache` directory. e.g. `cache/camera1/

Images need to be resized and converted to grayscale for the model. This can be performed using the `mogrify` tool in the ImageMagick software.

```
cd cache/[IMAGES_FOLDER]
mogrify -resize 1840x1228 *jpg
mogrify -type Grayscale *jpg
```

### Running the detection

Specify the folder name of images [`IMAGES_FOLDER`] stored in the `cache` directory

TODO: Kevin to update code to make this easier

Classify the images:

`python classify_images.py`

Generate annotated images that highlight tiles where transient objects are detected

`python parse_results.py`

Result files:

* CSV containing detection coordinates: `results/[IMAGES_FOLDER].csv`
* Annotated images with highlighted detection tiles: `results/[IMAGE_FOLDER]/`

### Training the model


