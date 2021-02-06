import numpy as np
cimport cython

#cython: boundscheck=False, wraparound=False

ctypedef fused floating:
    float
    double


cdef floating clip(floating value):
    return min(max(value, 0), 1)

def current_to_best_mutation(floating[:] current, floating[:] best, floating[:] x1, floating[:] x2, floating mut):
    cdef Py_ssize_t x_max = current.shape[0]

    cdef Py_ssize_t x
    cdef double temp
    result = np.zeros((x_max,), dtype = np.float_)

    for x in range(x_max):
        temp = current[x] + mut * (best[x] - current[x]) + mut * (x1[x] - x2[x])
        result[x] = clip(temp)

    return result


def rand_mutation(floating[:] x1, floating[:] x2, floating[:] x3, floating mut):

    cdef Py_ssize_t x_max = x1.shape[0]
    cdef Py_ssize_t x
    cdef double temp

    result = np.zeros((x_max,), dtype=np.float_)
    for x in range(x_max):
        temp = x1[x] + mut*(x2[x] - x3[x])
        result[x] = clip(temp)

    return result


