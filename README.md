# Fireballs

Data processing and analysis scripts for detecting transient objects (fireballs) from optical and radio astronomy images for the Curtin Institute for Radio Astronomy (CIRA).

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

### TensorFlow

The code base has been developing using TensorFlow 0.11.0 RC0. Note that TensorFlow 0.12.0 RC0 introduced [breaking changes to the API](https://github.com/tensorflow/tensorflow/releases/tag/0.12.0-rc0) and will not work with the existing code. 

Follow the below instructions to install the TensorFlow 0.11.0 RC0 version:

```
# Ubuntu/Linux 64-bit, CPU only, Python 2.7
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0rc0-cp27-none-linux_x86_64.whl

# Ubuntu/Linux 64-bit, GPU enabled, Python 2.7
# Requires CUDA toolkit 8.0 and CuDNN v5. 
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.11.0rc0-cp27-none-linux_x86_64.whl

# Mac OS X, CPU only, Python 2.7:
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.11.0rc0-py2-none-any.whl

# Mac OS X, GPU enabled, Python 2.7:
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/gpu/tensorflow_gpu-0.11.0rc0-py2-none-any.whl

# Ubuntu/Linux 64-bit, CPU only, Python 3.4
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0rc0-cp34-cp34m-linux_x86_64.whl

# Ubuntu/Linux 64-bit, GPU enabled, Python 3.4
# Requires CUDA toolkit 8.0 and CuDNN v5. 
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.11.0rc0-cp34-cp34m-linux_x86_64.whl

# Ubuntu/Linux 64-bit, CPU only, Python 3.5
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0rc0-cp35-cp35m-linux_x86_64.whl

# Ubuntu/Linux 64-bit, GPU enabled, Python 3.5
# Requires CUDA toolkit 8.0 and CuDNN v5. 
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.11.0rc0-cp35-cp35m-linux_x86_64.whl

# Mac OS X, CPU only, Python 3.4 or 3.5:
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.11.0rc0-py3-none-any.whl

# Mac OS X, GPU enabled, Python 3.4 or 3.5:
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/gpu/tensorflow_gpu-0.11.0rc0-py3-none-any.whl
```

Install TensorFlow:

```
# Python 2
$ sudo pip install --upgrade $TF_BINARY_URL

# Python 3
$ sudo pip3 install --upgrade $TF_BINARY_URL
```

## Usage

### Data pre-processsing
The code works on JPEG images. Images should be copied to a folder `[IMAGES_FOLDER]` within the `cache` directory. e.g. `cache/camera1/`

Images need to be resized and converted to grayscale to run the model. This can be performed using the `mogrify` tool in the ImageMagick software.

```
cd cache/[IMAGES_FOLDER]
mogrify -resize 1840x1228 *jpg
mogrify -type Grayscale *jpg
```

### Running the detection

Specify the folder name of images `[IMAGES_FOLDER]` stored in the `cache` directory within the `src/settings.py` file

```
# User defined settings
IMAGES_FOLDER_NAME = '[IMAGES_FOLDER]'
...
```

The model files are located within the `models` folder. The default detection model used is `models/transients`

#### Classify the images

`python classify_images.py`

Outputs:

* CSV containing detection coordinates: `results/[IMAGES_FOLDER].csv`
* 200x200 image tiles with detected transient objects : `output/[IMAGE_FOLDER]`


#### Annotated images to highlight where transient objects are detected

`python parse_results.py`

Outputs:

* Annotated images with highlighted detection tiles: `results/[IMAGE_FOLDER]/`


### Training the model

`python train_model.py`

### Generating synthetic datasets

`python create_dataset.py`
