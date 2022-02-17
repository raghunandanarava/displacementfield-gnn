import numpy as np
import numba as nb

@nb.jit(nopython=True)
def vrange(starts, stops, steps):
    discretised = []
    for i in range(len(starts)):
        discretised.append(np.arange(start=starts[i], stop=stops[i], step=steps))
    return np.asarray(np.column_stack([discretised, stops]))