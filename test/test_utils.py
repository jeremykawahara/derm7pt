from derm7pt import utils
import numpy as np


def test_strings2numeric():
    # Original strings to convert.
    strings = ['a', 'a', 'b', 'c', 'c']
    # The unique names of these strings.
    names = ['a', 'b', 'c']
    # And their corresponding numeric values to assign.
    numeric_vals = [0, 1, 2]
    # Expected output.
    exp_out = [0, 0, 1, 2, 2]
    out = utils.strings2numeric(strings, names, numeric_vals, sentinel=-1)
    assert np.alltrue(out == exp_out), "Error: unexpected numeric values returned."

    # Original strings to convert.
    strings = ['a', 'a', 'b', 'c', 'c', 'd']
    # The unique names of these strings.
    names = ['a', 'b', 'c', 'd']
    # And their corresponding numeric values to assign.
    numeric_vals = [0, 1, 2, 2]  # Note both 'c' and 'd' are assigned the numeric value of 2.
    # Expected output.
    exp_out = [0, 0, 1, 2, 2, 2]
    out = utils.strings2numeric(strings, names, numeric_vals, sentinel=-1)
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
    out = utils.strings2numeric(strings, names, numeric_vals, sentinel=-1)
    assert np.alltrue(out == exp_out), \
        "Error: unexpected numeric values returned when mapping lists to the same value."
