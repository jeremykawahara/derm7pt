# derm7pt
`derm7pt` preprocess the [Seven-Point Checklist Dermatolgoy Dataset](http://derm.cs.sfu.ca) and converts the data into a more acessible format. 

`derm7pt` is a Python module that serves as a starting point to use the data as described in,
> J. Kawahara, S. Daneshvar, G. Argenziano, and G. Hamarneh, “Seven-Point Checklist and Skin Lesion Classification using Multi-Task Multi-Modal Neural Nets,” IEEE Journal of Biomedical and Health Informatics, pp. 1–8, 2018. [[pdf]](http://www.cs.sfu.ca/~hamarneh/ecopy/jbhi2018a.pdf) [[doi]](https://doi.org/10.1109/JBHI.2018.2824327)

# Download the Data
http://derm.cs.sfu.ca

The images and meta-data (e.g., seven-point checklist criteria, diagnosis) can be downloaded from the external site above.

The actual images and meta-data are **not** stored in this repo. 

# Minimal Example
```python
import sys, os
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..'))) # To import derm7pt
from derm7pt.dataset import Derm7PtDatasetGroupInfrequent

# Change this line to your data directory.
dir_release = '/local-scratch/jer/data/argenziano/release_v0'

# Dataset after grouping infrequent labels.
derm_data = Derm7PtDatasetGroupInfrequent(
    dir_images=os.path.join(dir_release, 'images'), 
    metadata_df=pd.read_csv(os.path.join(dir_release, 'meta/meta.csv')), 
    train_indexes=list(pd.read_csv(os.path.join(dir_release, 'meta/train_indexes.csv'))['indexes']), 
    valid_indexes=list(pd.read_csv(os.path.join(dir_release, 'meta/valid_indexes.csv'))['indexes']), 
    test_indexes=list(pd.read_csv(os.path.join(dir_release, 'meta/test_indexes.csv'))['indexes']))

# Outputs to screen the preprocessed dataset in a Pandas format.
derm_data.df
```
This will group infrequent class labels together and assign numeric values to each class label.

You can see the output in [this minimal example notebook](https://github.com/jeremykawahara/derm7pt/blob/master/notebooks/minimal_example.ipynb).

You can find a more [comprehensive example here](https://github.com/jeremykawahara/derm7pt/blob/master/notebooks/example.ipynb) that includes an example of how to classify some of the seven-point checklist.

# Installation Instructions
You can see the dependencies and versions `derm7pt` was tested on [here](https://github.com/jeremykawahara/derm7pt/blob/master/version_check.ipynb).

To use `derm7pt`:
1. Download the [data](http://derm.cs.sfu.ca) and unzip it to your folder (we will use the folder `/local-scratch/jer/data/argenziano/release_v0` for this example)
2. Clone this repository
3. Run the [minimal_example.py](https://github.com/jeremykawahara/derm7pt/blob/master/minimal_example.py). Make sure to change the directory to match your data folder.

Steps #2 and #3 are shown below,
```
git clone https://github.com/jeremykawahara/derm7pt.git
cd derm7pt
python minimal_example.py '/local-scratch/jer/data/argenziano/release_v0'
```
This should output a view of the data that is similar to what is [shown in this notebook](https://github.com/jeremykawahara/derm7pt/blob/master/notebooks/minimal_example.ipynb).

# Related Publications

