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
The code works on jpeg images. If large, they need to be downsampled, for example using:
`convert in.jpg -resize 50% out.jpg`

### Running the detection

### Training the model
