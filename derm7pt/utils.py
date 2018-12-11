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


def _test_strings2numeric():
    # Original strings to convert.
    strings = ['a', 'a', 'b', 'c', 'c']
    # The unique names of these strings.
    names = ['a', 'b', 'c']
    # And their corresponding numeric values to assign.
    numeric_vals = [0, 1, 2]
    # Expected output.
    exp_out = [0, 0, 1, 2, 2]
    out = strings2numeric(strings, names, numeric_vals, sentinel=-1)
    assert np.alltrue(out == exp_out), "Error: unexpected numeric values returned."

    # Original strings to convert.
    strings = ['a', 'a', 'b', 'c', 'c', 'd']
    # The unique names of these strings.
    names = ['a', 'b', 'c', 'd']
    # And their corresponding numeric values to assign.
    numeric_vals = [0, 1, 2, 2]  # Note both 'c' and 'd' are assigned the numeric value of 2.
    # Expected output.
    exp_out = [0, 0, 1, 2, 2, 2]
    out = strings2numeric(strings, names, numeric_vals, sentinel=-1)
    assert np.alltrue(out == exp_out), \
        "Error: unexpected numeric values returned when multiple labels assigned same numeric value."

    # Original strings to convert.
    strings = ['a', 'a', 'b', 'c', 'c', 'd', 'bb', 'ba']
    # The unique names of these strings.
    names = ['a', ['b', 'bb', 'ba'], 'c', 'd']
    # And their corresponding numeric values to assign.
    numeric_vals = [0, 1, 2, 2]  # Note both 'c' and 'd' are assigned the numeric value of 2.
    # Expected output.
    exp_out = [0, 0, 1, 2, 2, 2, 1, 1]
    out = strings2numeric(strings, names, numeric_vals, sentinel=-1)
    assert np.alltrue(out == exp_out), \
        "Error: unexpected numeric values returned when mapping lists to the same value."


_test_strings2numeric()