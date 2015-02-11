import  cython
import  numpy as NP
cimport numpy as NP


DTYPE = NP.float64
ctypedef NP.float64_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def cy_calc_t2new(int nocc, int nspin, NP.ndarray[DTYPE_t, ndim=4] precomp_vee, NP.ndarray[DTYPE_t, ndim=1] orb_eng, NP.ndarray[DTYPE_t, ndim=4] t2, NP.ndarray[DTYPE_t, ndim=4] t2new):
    cdef int i, j, a, b
    cdef double orb_eng_diff
    for i in range(nocc):
        for j in range(i + 1, nocc):
            for a in range(nocc, nspin):
                for b in range(a + 1, nspin):
                    if abs(precomp_vee[i,a,j,b]) < 1.0e-8:
                        continue
                    orb_eng_diff = orb_eng[i] + orb_eng[j] - orb_eng[a] - orb_eng[b]
                    t2new[i,j,a,b] = cy_calc_numerator(i, j, a, b, nocc, nspin, precomp_vee, t2) / orb_eng_diff
                    t2new[j,i,a,b] = -t2new[i,j,a,b]
                    t2new[j,i,b,a] = t2new[i,j,a,b]
                    t2new[i,j,b,a] = -t2new[i,j,a,b]
    return None
    #return t2new


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def cy_calc_numerator(int i, int j, int a, int b, int nocc, int nspin, NP.ndarray[DTYPE_t, ndim=4] precomp_vee, NP.ndarray[DTYPE_t, ndim=4] t2):
    cdef double this_val
    cdef int k, l, c, d
    this_val = 0.0
    this_val += precomp_vee[i,a,j,b]
    for c in range(nocc, nspin):
        for d in range(nocc, nspin):
            this_val += precomp_vee[a,c,b,d] * t2[i,j,c,d] * 0.5
    for k in range(nocc):
        for l in range(nocc):
            this_val += precomp_vee[i,k,j,l] * t2[k,l,a,b] * 0.5
    for k in range(nocc):
        for c in range(nocc, nspin):
            this_val -= precomp_vee[b,c,k,j] * t2[i,k,a,c]
            this_val += precomp_vee[b,c,k,i] * t2[j,k,a,c]
            this_val += precomp_vee[a,c,k,j] * t2[i,k,b,c]
            this_val -= precomp_vee[a,c,k,i] * t2[j,k,b,c]
    for k in range(nocc):
        for l in range(nocc):
            for c in range(nocc, nspin):
                for d in range(nocc, nspin):
                    this_val += precomp_vee[k,c,l,d] * \
                                ((0.25 * t2[i,j,c,d] * t2[k,l,a,b]) \
                                - (0.5  * (t2[i,j,a,c] * t2[k,l,b,d]) + (t2[i,j,b,d] * t2[k,l,a,c])) \
                                - (0.5  * (t2[i,k,a,b] * t2[j,l,c,d]) + (t2[i,k,c,d] * t2[j,l,a,b])) \
                                + ((t2[i,k,a,c] * t2[j,l,b,d]) + (t2[i,k,b,d] * t2[j,l,a,c])))
    return this_val
    
