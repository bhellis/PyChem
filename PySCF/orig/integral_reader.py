import numpy as np
__author__ = 'benellis'


def read_smat(filename):
    """
    calls 'read_hcore' - refer to that documentation
    """
    return read_hcore(filename)


def read_hcore(filename):
    """
    reads an hcore AO integral file into a numpy matrix (rank=2)
    file should be formated as such:
    1:      N-AO
    2:      i   j   integralValue
    .
    .
    .
    EOF

    full numpy matrix is returned (no packing) with symmetry operations applied
    """
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if len(line) <= 1:
                size = int(line[0])
                hcore = np.zeros((size, size), dtype=np.float64)
            elif len(line) == 3:
                i, j, val = int(line[0])-1, int(line[1])-1, np.float64(line[2])
                hcore[i,j] = hcore[j,i] = val
    return hcore


def read_vee(filename):
    """
    reads a vee AO integral file into a numpy matrix (rank=4)
    file should be formated as such:
    1:      N-AO,   N-AO
    2:      i   j   k   l   integralValue
    .
    .
    .
    EOF

    full numpy matrix is returned (no packing) with symmetry operations applied
    """
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if len(line) <= 2:
                size1, size2 = int(line[0]), int(line[1])
                vee = np.zeros((size1, size1, size2, size2), dtype=np.float64)
            elif len(line) == 5:
                mu, nu, lmda, sgma, val = int(line[0]) - 1, int(line[1]) - 1, int(line[2]) - 1, int(line[3]) - 1,  np.float64(line[4])
                vee[mu,nu,lmda,sgma] = \
                vee[nu,mu,lmda,sgma] = \
                vee[mu,nu,sgma,lmda] = \
                vee[nu,mu,sgma,lmda] = \
                vee[lmda,sgma,mu,nu] = \
                vee[sgma,lmda,mu,nu] = \
                vee[lmda,sgma,nu,mu] = \
                vee[sgma,lmda,nu,mu] = \
                val
    return vee

def read_smat_pack(filename):
    """
    calls 'read_hcore' - refer to that documentation
    """
    return read_hcore(filename)


def read_hcore_pack(filename):
    """
    reads an hcore AO integral file into a numpy matrix (rank=2)
    file should be formated as such:
    1:      N-AO
    2:      i   j   integralValue
    .
    .
    .
    EOF

    full numpy matrix is returned upper-packed with symmetry operations applied
    """
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if len(line) <= 1:
                size = int(line[0])
                hcore = np.zeros((size, size), dtype=np.float64)
            elif len(line) == 3:
                i, j, val = int(line[0])-1, int(line[1])-1, np.float64(line[2])
                hcore[i,j] = hcore[j,i] = val
    return hcore


def read_vee_pack(filename):
    """
    reads a vee AO integral file into a numpy matrix (rank=4)
    file should be formated as such:
    1:      N-AO,   N-AO
    2:      i   j   k   l   integralValue
    .
    .
    .
    EOF

    numpy matrix is returned as upper-packed with symmetry operations applied
    """
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if len(line) <= 2:
                size1, size2 = int(line[0]), int(line[1])
                vee = np.zeros((size1, size1, size2, size2), dtype=np.float64)
            elif len(line) == 5:
                mu, nu, lmda, sgma, val = int(line[0]) - 1, int(line[1]) - 1, int(line[2]) - 1, int(line[3]) - 1,  np.float64(line[4])
                vee[mu,nu,lmda,sgma] = \
                vee[nu,mu,lmda,sgma] = \
                vee[mu,nu,sgma,lmda] = \
                vee[nu,mu,sgma,lmda] = \
                vee[lmda,sgma,mu,nu] = \
                vee[sgma,lmda,mu,nu] = \
                vee[lmda,sgma,nu,mu] = \
                vee[sgma,lmda,nu,mu] = \
                val
    return vee

