import sys, os
import pandas as pd
pd.set_option('display.max_columns', 5) # Show 5 columns for readability.
from derm7pt.dataset import Derm7PtDatasetGroupInfrequent

# Points to the directory that contains the data.
dir_release = sys.argv[1] # '/local-scratch/jer/data/argenziano/release_v0'

# Dataset after grouping infrequent labels.
derm_data = Derm7PtDatasetGroupInfrequent(
    dir_images=os.path.join(dir_release, 'images'), 
    metadata_df=pd.read_csv(os.path.join(dir_release, 'meta/meta.csv')), 
    train_indexes=list(pd.read_csv(os.path.join(dir_release, 'meta/train_indexes.csv'))['indexes']), 
    valid_indexes=list(pd.read_csv(os.path.join(dir_release, 'meta/valid_indexes.csv'))['indexes']), 
    test_indexes=list(pd.read_csv(os.path.join(dir_release, 'meta/test_indexes.csv'))['indexes']))

# Outputs to screen the first 5 rows of the preprocessed dataset in a Pandas format.
print(derm_data.df.head(n=5))