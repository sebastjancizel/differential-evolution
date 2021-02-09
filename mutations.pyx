import numpy as np
cimport cython
from cython.parallel cimport prange


ctypedef fused floating:
    float
    double

cdef floating clip(floating value) nogil:
    return min(max(value, 0), 1)

@cython.boundscheck(False)
@cython.wraparound(False)
def current_to_best_mutation(floating[:] current, floating[:] best, floating[:] x1, floating[:] x2, floating mut):

    cdef Py_ssize_t x_max = current.shape[0]
    cdef Py_ssize_t x
    cdef floating temp
    cdef floating[:] result = np.empty(x_max, dtype = np.float_)

    for x in prange(x_max,nogil=True):
        temp = current[x] + mut * (best[x] - current[x]) + mut * (x1[x] - x2[x])
        result[x] = clip(temp)

    return np.array(result)

@cython.boundscheck(False)
@cython.wraparound(False)
def rand_mutation(floating[:] x1, floating[:] x2, floating[:] x3, floating mut):

    cdef Py_ssize_t x_max = x1.shape[0]
    cdef Py_ssize_t x
    cdef floating temp
    cdef floating[:] result = np.empty(x_max, dtype=np.float_)

    for x in prange(x_max, nogil=True):
        temp = x1[x] + mut*(x2[x] - x3[x])
        result[x] = clip(temp)

    return np.array(result)


