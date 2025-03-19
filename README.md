# CDR Profiler

The CDR - Profiler is an innovative software tool designed to extend and enhance the traditional Cup-to-Disc Ratio (CDR) measurement used in ophthalmic imaging. Unlike conventional methods that typically provide a single vertical or horizontal CDR value, this tool offers a comprehensive profile of CDR values around the entire optic disc.

## Installation

We'll use *Anaconda* to create a new Python environment and handle all the required dependencies.

1. Install Anaconda following the [official guidelines](https://docs.anaconda.com/anaconda/install/).

2. Clone this repo to your machine.

3. Open a new Anaconda terminal (Windows) or a normal terminal window (Linux/MacOS) and cd to the directory of the cloned repo: `cd <repo_directory>`

4. Open a new Anaconda terminal and create a new environment: `conda create -n pcdr python=3.12`

5. Activate your newly create environment: `conda activate pcdcr`

6. Install the required libraries to run CDR-Profiler: `pip install -r requirements.txt`

7. Install [Pytorch](https://pytorch.org/get-started/locally/) using `pip`

## Usage

1. Modify the configuration parameters found in `cfg/config.ini` to suit your needs.

2. Download the model weights from [here](https://github.com/Borja21091/CDR-Profiler/releases/tag/v1.0) (disc.pth, cup.pth, fovea.pth) and place those files in `src/models`

3. Place all your images in the input folder you've set in the previous step. The software uses `data/` as the default folder to look for input images.

4. Run `main.py`

5. Check the ouput folder (default is `results/`) for the results. The folder should contain a file called `results.csv` with all the measurements. Images with the segmentations and CDR-Profiles will also be saved there if saving the result image is set as `True` in the config file.

### Disclaimer

Current implementation uses automatic segmentation of fovea, disc and cup. In future releases, the software will accept user generated masks.
