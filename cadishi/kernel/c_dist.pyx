# Copyright 2015 Juergen Koefinger

import numpy as np
cimport numpy as np
import cython

cdef extern from "math.h":
    double sqrt(double t)

@cython.wraparound(False)
@cython.boundscheck(False)

def pwd(np.ndarray[np.double_t, ndim=2] r, np.ndarray[np.double_t, ndim=1] histo, double rmax, int nbins):
#Calculate histogram of distances between particles belonging to the same structure (=set of coordinates).
    cdef int i, j, k, rIndex
    cdef np.double_t tmp, d
    for i in range(r.shape[0]-1):
        for j in range(i+1, r.shape[0]):
            d=0.
            for k in range(3):
                tmp=r[i,k]-r[j,k]
                d+=tmp*tmp
            d=sqrt(d)
            #d=sqrt(((r[i] - r[j])**2).sum())
            rIndex=int(d/rmax*nbins)
            if rIndex>=nbins:
                print "\n rIndex>=nbins\n"
            histo[rIndex]+=1
    return 

def pwd2(np.ndarray[np.double_t, ndim=2] r1, np.ndarray[np.double_t, ndim=2] r2, np.ndarray[np.double_t, ndim=1] histo, double rmax, int nbins):
#Calculate histogram of distances between particles belonging to different structures(=sets of coordinates).
    cdef int i, j, k, rIndex
    cdef np.double_t tmp, d
    for i in range(r1.shape[0]):
        for j in range(r2.shape[0]):
            d=0.
            for k in range(3):
                tmp=r1[i,k]-r2[j,k]
                d+=tmp*tmp
            d=sqrt(d)
            rIndex=int(d/rmax*nbins)
            if rIndex>=nbins:
                print "\n rIndex>=nbins\n"
            histo[rIndex]+=1
    return 

def pwdNoRoot(np.ndarray[np.double_t, ndim=2] r, np.ndarray[np.double_t, ndim=1] histo, double rmax, int nbins):
#Calculate histogram of distances between particles belonging to the same structure (=set of coordinates).
    cdef int i, j, k, rIndex
    cdef np.double_t tmp, d
    for i in range(r.shape[0]-1):
        for j in range(i+1, r.shape[0]):
            d=0.
            for k in range(3):
                tmp=r[i,k]-r[j,k]
                d+=tmp*tmp
            rIndex=int(d/rmax*nbins)
            histo[rIndex]+=1
    return 
