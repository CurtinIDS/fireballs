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

Specify the folder name of images `[IMAGES_FOLDER]` stored in the `cache` directory within the `src/settings.py` file by updating the `IMAGES_FOLDER_NAME` variable.

```
# User defined settings
IMAGES_FOLDER_NAME = '[IMAGES_FOLDER]'
MODEL_FILE_NAME = 'transients'
...
```

Model files are located within the `models` folder. The default detection model is `models/transients` but a different model can be used by updating the `MODEL_FILE_NAME` variable in `src/settings.py`.

#### Classify the images

```
cd src
python classify_images.py
```

Outputs:

* CSV file containing detection coordinates: `results/[IMAGES_FOLDER].csv`
* 200x200 image tiles with detected transient objects : `output/[IMAGE_FOLDER]`


#### Annotated images by highlighting where objects are detected

```
cd src
python parse_results.py
```

Outputs:

* Annotated images: `results/[IMAGE_FOLDER]/`


### Training the model

#### Prepare the training dataset

A training and validation dataset of images must be provided for training the model. The dataset folder `[TRAINING_DATASET_FOLDER]` and following subfolders must be created within the `data` directory. 

```
data/
  [TRAINING_DATASET_FOLDER]/
    training/
      0/
      1/
    validation/
      0/
      1/
```

Notes:

* `0/` images that do NOT contain any transient objects
* `1/` images with transient objects
* images must be:
 * 200x200 pixels
 * grayscale
* a validation dataset that is 10% the size of the training dataset was used in experiments  

Specify the folder name of the training images stored within `src/settings.py` file by updating the `TRAINING_DATASET_FOLDER` variable.

`TRAINING_DATASET_FOLDER = '[TRAINING_DATASET_FOLDER]'`

#### Training

Specify conditions for training the model by updating variables in the `src/train_model.py` file. 

```
# Incrementally train from an existing model
LOAD_EXISTING_MODEL = False
EXISTING_MODEL = s.MODELS_FOLDER + 'experiment/exp5'
# New trained model name
MODEL_NAME = 'synthetic'
# Store model checkpoint files
CHECKPOINT_FOLDER = s.OUTPUT_FOLDER + MODEL_NAME
```

Train the model

`python train_model.py`

Notes:

* Setting `LOAD_EXISTING_MODEL = True` and providing a `EXISTING_MODEL` file allows the model to be traing using the weights of an existing model rather than training from scratch
* Trained model checkpoint files are saved to `output/[MODEL_NAME]`
 * copy the last checkpoint model file to `models/` if you wish to use it for classifying images
  * e.g. `cp output/[MODEL_NAME]/-5032 models/[NEW_MODEL_NAME]`
  * `MODEL_FILE_NAME = '[NEW_MODEL_NAME]'` should then be specified within `settings.py`


### Generating synthetic datasets

A dataset of background images containing no transient objects must be provided to generate a synthetic dataset for training the model. Create folders `[SYNTHETIC_DATASET_FOLDER]` and `[SYNTHETIC_DATASET_FOLDER]/source` within the `data` directory and copy the background images to the `source` folder. 

Modify variables in `create_dataset.py` to adjust how the dataset is created:
* `TRAINING_SAMPLES`
* `VALIDATION_SAMPLES`
* `BIAS`
* `BRIGHTNESS_VALUES`
* `BRIGHTNESS_THRESHOLD_VALUES`

Generate the synthetic dataset:

`python create_dataset.py`

If you want to use the synthetically generated dataset for training the model then modify `settings.py` and uncomment `# TRAINING_DATASET_FOLDER = SYNTHETIC_DATASET_FOLDER`.


### Visualising the model

`jupyter-notebook src/visualise_cnn.ipynb`
