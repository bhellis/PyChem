import numpy as np
import math
from scipy import linalg
from PySCF import chem_sys


def read_vnuc(filename):
    with open(filename) as f:
        lines = f.readlines()
    natom = len(lines)
    xyz  = np.zeros((natom, 3), dtype = np.float64)
    znuc = np.zeros((3), dtype = np.float64)
    for i in range(0, len(lines)):
        line = lines[i].strip().split()
        znuc[i], xyz[i][0], xyz[i][1], xyz[i][2] = float(line[0]), float(line[1]), float(line[2]), float(line[3])
    return natom, znuc, xyz


def calc_vnuc(filename):
    natom, znuc, xyz = read_vnuc(filename)
    vnuc = 0.0
    for i in range(natom):
        for j in range(i + 1, natom):
            rij = math.sqrt(((xyz[i][0] - xyz[j][0]) * (xyz[i][0] - xyz[j][0])) + \
                            ((xyz[i][1] - xyz[j][1]) * (xyz[i][1] - xyz[j][1])) + \
                            ((xyz[i][2] - xyz[j][2]) * (xyz[i][2] - xyz[j][2])))
            vnuc += (znuc[i] * znuc[j]) / rij
    return vnuc


def calc_pmat(sys_param, fmat, smat):
    evec, cvec = linalg.eigh(fmat, smat)
    nspace = sys_param.get_nspace()
    nocc   = sys_param.get_nocc()
    pmat = np.zeros((nspace, nspace), dtype=np.float64)
    for mu in range(nspace):
        for nu in range(nspace):
            for a in range(nocc // 2): 
                pmat[mu,nu] += cvec[mu,a] * cvec[nu,a] * 2.0
    return pmat


def calc_fock(sys_param, hcore, vee, pmat):
    nspace = sys_param.get_nspace()
    nocc   = sys_param.get_nocc()
    fmat = np.zeros((nspace, nspace), dtype=np.float64)
    for mu in range(nspace):
        for nu in range(nspace):
            thissum = 0.0
            for lmda in range(nspace):
                for sgma in range(nspace):
                    thissum += (pmat[lmda,sgma] * (vee[mu,nu,sgma,lmda] - (0.5*(vee[mu,lmda,sgma,nu]))))
            fmat[mu,nu] = hcore[mu,nu] + thissum
    return fmat


def calc_energy(sys_param, hcore, fmat, pmat):
    nspace = sys_param.get_nspace()
    eng = 0.0
    for mu in range(nspace):
        for nu in range(nspace):
            eng += pmat[nu,mu] * (hcore[mu,nu] + fmat[mu,nu])
    return 0.5 * eng















    
    
