import numpy as np
import pandas as pd
from derm7pt.utils import strings2numeric


class ArgenzianoDataset(object):
    # 'names': the name of the tag associated with the image.
    # 'abbrevs': a unique abbreviation. Used as the key to link other DataFrames.
    # 'colnames': the name of the column in the CSV that corresponds to this tag.
    # 'seven_pt': 1 if part of the 7-point criteria. Else 0 (diagnosis get's 0).

    # These are the `categories` (i.e., diagnosis + 7pt checklist) from the JBHI paper.
    tags = pd.DataFrame([
        {'names': 'Diagnosis', 'abbrevs': 'DIAG', 'colnames': 'diagnosis', 'seven_pt': 0},
        {'names': 'Pigment Network', 'abbrevs': 'PN', 'colnames': 'pigment_network', 'seven_pt': 1},
        {'names': 'Blue Whitish Veil', 'abbrevs': 'BWV', 'colnames': 'blue_whitish_veil', 'seven_pt': 1},
        {'names': 'Vascular Structures', 'abbrevs': 'VS', 'colnames': 'vascular_structures', 'seven_pt': 1},
        {'names': 'Pigmentation', 'abbrevs': 'PIG', 'colnames': 'pigmentation', 'seven_pt': 1},
        {'names': 'Streaks', 'abbrevs': 'STR', 'colnames': 'streaks', 'seven_pt': 1},
        {'names': 'Dots and Globules', 'abbrevs': 'DaG', 'colnames': 'dots_and_globules', 'seven_pt': 1},
        {'names': 'Regression Structures', 'abbrevs': 'RS', 'colnames': 'regression_structures', 'seven_pt': 1},
    ])

    # Each `category` has several `labels` associated with it. Each `label` has several properties.
    # `nums`: an integer unique within a category, used for the neural network.
    # `names`: A string that corresponds to the values within the csv file that represent the type of label.
    #       Multiple labels can be grouped to form a single label by passing in a list of strings.
    # `abbrevs`: a unique abbreviation that represents the label.
    # `info`: include other helpful info for yourself here. Not otherwise used.
    #         I kept forgetting what some diseases were, so included additional info for a few.
    # Note there is no `scores` since the diagnosis is an overall label of the image,
    # and does not contribute to the 7-point checklist score.
    diagnosis = pd.DataFrame([
        {'nums': 0, 'names': 'basal cell carcinoma', 'abbrevs': 'BCC', 'info': 'Common non-melanoma cancer'},
        {'nums': 1, 'names': 'blue nevus', 'abbrevs': 'BLN'},
        {'nums': 2, 'names': 'clark nevus', 'abbrevs': 'CN'},
        {'nums': 3, 'names': 'combined nevus', 'abbrevs': 'CBN'},
        {'nums': 4, 'names': 'congenital nevus', 'abbrevs': 'CGN'},
        {'nums': 5, 'names': 'dermal nevus', 'abbrevs': 'DN'},
        {'nums': 6, 'names': 'dermatofibroma', 'abbrevs': 'DF'},
        {'nums': 7, 'names': 'lentigo', 'abbrevs': 'LT'},
        {'nums': 8, 'names': ['melanoma', 'melanoma (in situ)', 'melanoma (less than 0.76 mm)',
                              'melanoma (0.76 to 1.5 mm)', 'melanoma (more than 1.5 mm)',
                              'melanoma metastasis'], 'abbrevs': 'MEL'},
        {'nums': 9, 'names': 'melanosis', 'abbrevs': 'MLS', 'info': 'Hyperpigmentation of the skin.'},
        {'nums': 10, 'names': 'miscellaneous', 'abbrevs': 'MISC'},
        {'nums': 11, 'names': 'recurrent nevus', 'abbrevs': 'RN'},
        {'nums': 12, 'names': 'reed or spitz nevus', 'abbrevs': 'RSN'},
        {'nums': 13, 'names': 'seborrheic keratosis', 'abbrevs': 'SK'},
        {'nums': 14, 'names': 'vascular lesion', 'abbrevs': 'VL'},
    ])

    # `scores`: An integer that represents how much the label contributes to
    #           the 7-point checklist score.
    pigment_network = pd.DataFrame([
        {'nums': 0, 'names': 'absent', 'abbrevs': 'ABS', 'scores': 0, 'info': ''},
        {'nums': 1, 'names': 'typical', 'abbrevs': 'TYP', 'scores': 0, 'info': ''},
        {'nums': 2, 'names': 'atypical', 'abbrevs': 'ATP', 'scores': 2, 'info': ''},
    ])

    blue_whitish_veil = pd.DataFrame([
        {'nums': 0, 'names': 'absent', 'abbrevs': 'ABS', 'scores': 0, 'info': ''},
        {'nums': 1, 'names': 'present', 'abbrevs': 'PRS', 'scores': 2, 'info': ''},
    ])

    vascular_structures = pd.DataFrame([
        {'nums': 0, 'names': 'absent', 'abbrevs': 'ABS', 'scores': 0, 'info': ''},
        {'nums': 1, 'names': 'arborizing', 'abbrevs': 'ARB', 'scores': 0, 'info': ''},
        {'nums': 2, 'names': 'comma', 'abbrevs': 'COM', 'scores': 0, 'info': ''},
        {'nums': 3, 'names': 'hairpin', 'abbrevs': 'HP', 'scores': 0, 'info': ''},
        {'nums': 4, 'names': 'within regression', 'abbrevs': 'WR', 'scores': 0, 'info': ''},
        {'nums': 5, 'names': 'wreath', 'abbrevs': 'WTH', 'scores': 0, 'info': ''},
        {'nums': 6, 'names': 'dotted', 'abbrevs': 'DOT', 'scores': 2, 'info': ''},
        {'nums': 7, 'names': 'linear irregular', 'abbrevs': 'LIR', 'scores': 2, 'info': ''},
    ])

    pigmentation = pd.DataFrame([
        {'nums': 0, 'names': 'absent', 'abbrevs': 'ABS', 'scores': 0, 'info': ''},
        {'nums': 1, 'names': 'diffuse regular', 'abbrevs': 'DR', 'scores': 0, 'info': ''},
        {'nums': 2, 'names': 'localized regular', 'abbrevs': 'LR', 'scores': 0, 'info': ''},
        {'nums': 3, 'names': 'diffuse irregular', 'abbrevs': 'DI', 'scores': 1, 'info': ''},
        {'nums': 4, 'names': 'localized irregular', 'abbrevs': 'LI', 'scores': 1, 'info': ''},
    ])

    streaks = pd.DataFrame([
        {'nums': 0, 'names': 'absent', 'abbrevs': 'ABS', 'scores': 0, 'info': ''},
        {'nums': 1, 'names': 'regular', 'abbrevs': 'REG', 'scores': 0, 'info': ''},
        {'nums': 2, 'names': 'irregular', 'abbrevs': 'IR', 'scores': 1, 'info': ''},
    ])

    dots_and_globules = pd.DataFrame([
        {'nums': 0, 'names': 'absent', 'abbrevs': 'ABS', 'scores': 0, 'info': ''},
        {'nums': 1, 'names': 'regular', 'abbrevs': 'REG', 'scores': 0, 'info': ''},
        {'nums': 2, 'names': 'irregular', 'abbrevs': 'IR', 'scores': 1, 'info': ''},
    ])

    regression_structures = pd.DataFrame([
        {'nums': 0, 'names': 'absent', 'abbrevs': 'ABS', 'scores': 0, 'info': ''},
        {'nums': 1, 'names': 'blue areas', 'abbrevs': 'BA', 'scores': 1, 'info': ''},
        {'nums': 2, 'names': 'white areas', 'abbrevs': 'WA', 'scores': 1, 'info': ''},
        {'nums': 3, 'names': 'combinations', 'abbrevs': 'CMB', 'scores': 1, 'info': ''},
    ])

    def __init__(self, dir_images, metadata_df, train_indexes, valid_indexes, test_indexes, crop_amount=25):
        """The meta-data for the Argenziano dataset.

        Args:
            dir_images: A string indicating the root directory of the images.
            metadata_df: A Pandas data-frame that contains all the meta-data for each case.
            train_indexes: A list of integers that represent training indexes into metadata_df.
            valid_indexes: 
            test_indexes:
            crop_amount: An integer specifying how many pixels to crop at the image border.
                Useful if images contain a black boundary.
        """

        self.derm_column = 'derm'
        self.clinic_column = 'clinic'

        self.df = metadata_df
        self.dir_imgs = dir_images
        self.crop_amount = crop_amount

        # Modify meta to include numeric labels in the columns.
        self.set_df_numeric_labels()

        # Check the properties of the class match the columns in self.df
        self.check_myself()

        # Make sure all the indexes are in at least one fold.
        match_indexes = np.alltrue(np.sort(np.concatenate((train_indexes, valid_indexes, test_indexes)))
                                   == range(len(self.df)))
        if not match_indexes:
            print("Warning! The train/valid/test indexes do not match the total number of samples.")

        all_indexes = np.concatenate((train_indexes, valid_indexes, test_indexes))
        assert len(set(all_indexes)) == len(all_indexes), "Error! There are duplicate indexes in train, valid, or test."

        self.train = self.df.iloc[train_indexes]
        self.valid = self.df.iloc[valid_indexes]
        self.test = self.df.iloc[test_indexes]

        # Meta data.
        self.elevation_dict = self.get_dict_labels(self.df.elevation)
        self.sex_dict = self.get_dict_labels(self.df.sex)
        self.location_dict = self.get_dict_labels(self.df.location)

    def set_df_numeric_labels(self):
        """Add numeric values to the columns in the df."""
        for abbrev in self.tags.abbrevs:
            col_name = self.get_column_name(abbrev)
            nums = strings2numeric(self.df[col_name], self.get_label_names(abbrev), self.get_label_nums(abbrev))
            col_name_numeric = self.get_column_name_numeric(abbrev)
            self.df[col_name_numeric] = nums

    def get_column_name(self, abbrev):
        """Return the data-frame column name that corresponds to the string labels for this `abbrev`."""
        tag = self.get_tag_by_abbrev(abbrev)
        return tag.colnames.iloc[0]

    def get_tag_by_abbrev(self, abbrev):
        """Return the label info for a given abbreviation."""
        tag = self.tags[self.tags.abbrevs == abbrev]
        if tag.empty:
            raise ValueError('Error: no tag for `%s`' % str(abbrev))

        return tag

    def get_label_names(self, abbrev, ignore_sub_names=False):
        """Return the names for all the labels for a given category abbrev."""
        lab = self.get_label_by_abbrev(abbrev)
        label_names = []
        for name in lab.names.values:
            if ignore_sub_names:
                if type(name) is list:
                    # Use the first name in the list if is a list.
                    name = name[0]

            label_names.append(name)

        return label_names

    def get_label_by_abbrev(self, abbrev):
        tag = self.get_tag_by_abbrev(abbrev)
        lab = getattr(self, tag.colnames.values[0])
        return lab

    def get_label_nums(self, abbrev):
        lab = self.get_label_by_abbrev(abbrev)
        return lab.nums.values

    def get_column_name_numeric(self, abbrev):
        """Return the name of the data-frame column that corresponds to the numeric labels for this `abbrev`."""
        col_name = self.get_column_name(abbrev)
        return col_name + '_numeric'

    @staticmethod
    def get_dict_labels(df_column_names):

        label_names = df_column_names.unique()
        label_names.sort()
        label_dict = {}
        for label_idx, label_name in enumerate(label_names):
            label_dict[label_name] = label_idx

        return label_dict

    def check_myself(self):
        """Check the properties of the class match the columns in self.df"""
        for var_name in self.tags.colnames:

            # Check the column names correspond to the attributes of the class.
            var = getattr(self, var_name, None)
            if var is None:
                raise ValueError('Error: missing attribute `self.%s`. Make sure you manually set it.' % str(var_name))

            # Check that all the column names correspond to a column name in the CSV.
            if var_name not in self.df.columns:
                raise ValueError('Error: the variable name `%s` does not link to a column name.' % str(var_name))


class ArgenzianoDatasetGroupInfrequent(ArgenzianoDataset):
    diagnosis = pd.DataFrame([
        {'nums': 0, 'names': 'basal cell carcinoma', 'abbrevs': 'BCC', 'info': 'Common non-melanoma cancer'},
        {'nums': 1,
         'names': ['nevus', 'blue nevus', 'clark nevus', 'combined nevus', 'congenital nevus', 'dermal nevus',
                   'recurrent nevus', 'reed or spitz nevus'], 'abbrevs': 'NEV'},
        {'nums': 2,
         'names': ['melanoma', 'melanoma', 'melanoma (in situ)', 'melanoma (less than 0.76 mm)',
                   'melanoma (0.76 to 1.5 mm)',
                   'melanoma (more than 1.5 mm)', 'melanoma metastasis'], 'abbrevs': 'MEL'},
        {'nums': 3, 'names': ['DF/LT/MLS/MISC', 'dermatofibroma', 'lentigo', 'melanosis',
                              'miscellaneous', 'vascular lesion'], 'abbrevs': 'MISC'},
        {'nums': 4, 'names': 'seborrheic keratosis', 'abbrevs': 'SK'},
    ])

    vascular_structures = pd.DataFrame([
        {'nums': 0, 'names': 'absent', 'abbrevs': 'ABS', 'scores': 0, 'info': ''},
        {'nums': 1, 'names': ['regular', 'arborizing', 'comma', 'hairpin', 'within regression', 'wreath'],
         'abbrevs': 'REG', 'scores': 0, 'info': ''},
        {'nums': 2, 'names': ['dotted/irregular', 'dotted', 'linear irregular'], 'abbrevs': 'IR', 'scores': 2,
         'info': ''},
    ])

    pigmentation = pd.DataFrame([
        {'nums': 0, 'names': 'absent', 'abbrevs': 'ABS', 'scores': 0, 'info': ''},
        {'nums': 1, 'names': ['regular', 'diffuse regular', 'localized regular'], 'abbrevs': 'REG', 'scores': 0,
         'info': ''},
        {'nums': 2, 'names': ['irregular', 'diffuse irregular', 'localized irregular'], 'abbrevs': 'IR', 'scores': 1,
         'info': ''},
    ])

    regression_structures = pd.DataFrame([
        {'nums': 0, 'names': 'absent', 'abbrevs': 'ABS', 'scores': 0, 'info': ''},
        {'nums': 1, 'names': ['present', 'blue areas', 'white areas', 'combinations'], 'abbrevs': 'PRS', 'scores': 1,
         'info': ''},
    ])
