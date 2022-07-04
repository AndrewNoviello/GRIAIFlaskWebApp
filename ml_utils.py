import numpy as np
import math

def truncate_float_array(xs, precision=3):
    return [truncate_float(x, precision=precision) for x in xs]


def truncate_float(x, precision=3):
    assert precision > 0

    if np.isclose(x, 0):
        return 0
    else:
        # Determine the factor, which shifts the decimal point of x
        # just behind the last significant digit
        factor = math.pow(10, precision - 1 - math.floor(math.log10(abs(x))))
        # Shift decimal point by multiplicatipon with factor, flooring, and
        # division by factor
        return math.floor(x * factor)/factor