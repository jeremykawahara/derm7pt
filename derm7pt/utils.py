import numpy as np


def strings2numeric(strings, names, numeric_vals, sentinel=-1):
    """Convert strings to numeric values.

    Args:
        strings: a list of strings to convert to numeric tags.
        names: a list of unique strings that correspond to the numeric values.
        numeric_vals: a list of integers that correspond to the ordering in names.
        sentinel: a value that is not in the numeric_vals.

    Returns:
        a numpy array of the numeric values to use (instead of the strings)

    """

    if sentinel in numeric_vals:
        raise ValueError("`sentinel` should not occur in `numeric_val`.")

    strings = np.asarray(strings)

    numeric = np.ones(shape=len(strings), dtype=int) * sentinel

    for class_idx, label in zip(numeric_vals, names):
        # If this is a list, then group all these (sublist) items with the same numeric label.
        if isinstance(label, list):
            for l in label:
                numeric[np.asarray(strings == l)] = class_idx

        else:  # This is a single sub-label already.
            numeric[np.asarray(strings == label)] = class_idx

    if np.any(numeric == sentinel):
        missing_indexes = np.where(numeric == sentinel)
        first_missing_str = strings[missing_indexes[0]]
        raise ValueError(
            "The value `%s` in `strings` do not exist in `names`. Did you spell something wrong?" % first_missing_str)

    return numeric


def html_image_src(image_paths, n_columns=4):
    """Print to screen twitter bootstrap html code for the `image_paths`.

    Args:
        image_paths: List of strings indicating the image paths.
        n_columns: Integer indicating the number of columns to divide the image by.
            Note that twitter bootstrap assumes 12 columns are available.
            So if you change `n_columns` you likely need to update `col-3` so n_columns * col-x = 12.

    """
    n_images = len(image_paths)
    n_rows = int(np.ceil(n_images / n_columns))

    image_src = ''
    column_count = 0
    for img_name in image_paths:
        if (column_count % n_rows) == 0:
            image_src += '<div class="col-3">\n'

        image_src += '  <img src="' + img_name + '" class="img-thumbnail" alt="' + img_name + '" onclick="window.open(this.src)">\n'

        column_count += 1

        if (column_count % n_rows) == 0:
            image_src += '</div>\n'

    image_src += '</div>\n'
    return image_src
