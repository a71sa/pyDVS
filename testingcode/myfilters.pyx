cimport cython
import numpy as np
cimport numpy as np

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.uint8

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.uint8_t DTYPE_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef winnumcheck(np.ndarray[DTYPE_t, ndim=2]  ar):
    cdef int i,j
    cdef int k=0
    for i in range(0,3):
        for j in range(0,3):
            if ar[i,j]>0:
                k+=1
    return k





@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def singlenoisefilter(np.ndarray[DTYPE_t, ndim=2]  pic2d,int size,int thresh):
    cdef int i,j
    for i in range(1,size-1):
        for j in range(1,size-1):
            if np.count_nonzero(pic2d[i-1:i+2,j-1:j+2])>=thresh:
                pass
            else :
                pic2d[i,j]=0
    pic2d[0,:]=0
    pic2d[:,0]=0
    pic2d[size-1,:]=0
    pic2d[:,size-1]=0