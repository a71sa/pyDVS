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
def singlenoisefilter(np.ndarray[DTYPE_t, ndim=2] flag,np.ndarray[DTYPE_t, ndim=2]  pic2d,int size,int thresh):
    cdef int i,j
    cdef int x,y
    cdef int k=0
    
    for i in range(1,size-1):
        for j in range(1,size-1):
            k=0
            for x in range(0,3):
                for y in range(0,3):
                    if pic2d[i-1+x,j-1+y]>0:
                        k+=1
            if(k>=thresh):
                for x in range(0,3):
                    for y in range(0,3):
                        flag[i-1+x,j-1+y]=1
            else :
                if flag[i,j]==0:
                    pic2d[i,j]=0

    pic2d[0,:]=0
    pic2d[:,0]=0
    pic2d[size-1,:]=0
    pic2d[:,size-1]=0



#=================================
