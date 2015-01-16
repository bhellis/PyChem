import math
import numpy as NP


class Chem_sys(object):
    def __init__(self, filename):
        self.nocc, self.nspace, self.pmat_cutoff, scale_hcore, scale_vee  = _read_input(filename)
        self.e_ref = 0.0
        self.vnuc = 0.0
        self.e_corr = 0.0

        if self.nocc % 2 != 0:
            print('Error --> This is an RHF code, Nocc must be even --> Exiting...')
            sys.exit()

        self.nspin = 2 * self.nspace
        self.hcore = scale_hcore * _read_hcore('hcore1.fmt')
        self.vee   = scale_vee * _read_vee('vee1.fmt')
        self.smat  = _read_smat('smat1.fmt')
        self.vnuc  = _calc_vnuc('geom1.fmt')

    def show_e_info(self, method):
        print('-- %s Energy Info --' % method)
        print('ElecRef: %s    ElecCorr: %s    Vnuc: %s    Total: %s' % (self.e_ref, self.e_corr, self.vnuc, (self.e_ref + self.e_corr + self.vnuc)))

    def get_vnuc(self):
        return self.vnuc

    def get_smat(self):
        return self.smat

    def get_vee(self):
        return self.vee

    def get_hcore(self):
        return self.hcore

    def get_nocc(self):
        return self.nocc
    
    def get_nspin(self):
        return self.nspin
    
    def get_nspace(self):
        return self.nspace
    
    def get_pmat_cutoff(self):
        return self.pmat_cutoff

    def set_hcore(self, hcorein):
        self.hcore = hcorein

    def set_vee(self, veein):
        self.vee = veein

    def set_e_ref(self, ein):
        self.e_ref = ein

    def set_e_corr(self, e_corr):
        self.e_corr = e_corr

    def get_hcore(self):
        return self.hcore

    def get_vee(self):
        return self.vee

    def set_orb_eng(self, orb_eng):
        self.orb_eng = orb_eng

    def get_orb_eng(self):
        return self.orb_eng

def _read_input(filename):
    with open(filename) as f:
        lines = f.readlines()
    thisline = lines[0].split()
    nocc         = int(thisline[1])
    thisline = lines[1].split()
    nspace       = int(thisline[1])
    thisline = lines[2].split()
    pmat_cutoff  = float(thisline[1])
    thisline = lines[3].split()
    scale_hcore  = float(thisline[1])
    thisline = lines[4].split()
    scale_vee    = float(thisline[1])
    return nocc, nspace, pmat_cutoff, scale_hcore, scale_vee


def _read_smat(filename):
    """
    calls 'read_hcore' - refer to that documentation
    """
    return _read_hcore(filename)


def _read_hcore(filename):
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
                hcore = NP.zeros((size, size), dtype=NP.float64)
            elif len(line) == 3:
                i, j, val = int(line[0])-1, int(line[1])-1, NP.float64(line[2])
                hcore[i,j] = hcore[j,i] = val
    return hcore


def _read_vee(filename):
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
                vee = NP.zeros((size1, size1, size2, size2), dtype=NP.float64)
            elif len(line) == 5:
                mu, nu, lmda, sgma, val = int(line[0]) - 1, int(line[1]) - 1, int(line[2]) - 1, int(line[3]) - 1,  NP.float64(line[4])
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

def _read_vnuc(filename):
    with open(filename) as f:
        lines = f.readlines()
    natom = len(lines)
    xyz  = NP.zeros((natom, 3), dtype = NP.float64)
    znuc = NP.zeros((3), dtype = NP.float64)
    for i in range(0, len(lines)):
        line = lines[i].strip().split()
        znuc[i], xyz[i][0], xyz[i][1], xyz[i][2] = float(line[0]), float(line[1]), float(line[2]), float(line[3])
    return natom, znuc, xyz


def _calc_vnuc(filename):
    natom, znuc, xyz = _read_vnuc(filename)
    vnuc = 0.0
    for i in range(natom):
        for j in range(i + 1, natom):
            rij = math.sqrt(((xyz[i][0] - xyz[j][0]) * (xyz[i][0] - xyz[j][0])) + \
                            ((xyz[i][1] - xyz[j][1]) * (xyz[i][1] - xyz[j][1])) + \
                            ((xyz[i][2] - xyz[j][2]) * (xyz[i][2] - xyz[j][2])))
            vnuc += (znuc[i] * znuc[j]) / rij
    return vnuc
