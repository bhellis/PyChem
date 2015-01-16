# installed
import sys
import math
import numpy as NP
from scipy import linalg as LA

# user
#from PySCF import scf_routines as SCF
#from PySCF import integral_reader as IntRd
from ChemSys import chem_sys
from PySCF import transforms as tran


def driver_scf(chem1):
    nocc   = chem1.get_nocc()
    nspace = chem1.get_nspace()
    nspin  = chem1.get_nspin()
    hcore  = chem1.get_hcore()
    vee    = chem1.get_vee()
    smat   = chem1.get_smat()
    vnuc   = chem1.get_vnuc()

    # get guess density / energy
    pmat = calc_pmat(chem1, hcore, smat)
    fmat = calc_fock(chem1, hcore, vee, pmat)
    e_hf = calc_energy(chem1, hcore, fmat, pmat)
    norm_last = LA.norm(pmat)

    # SCF cycle
    ncycle = 0
    while True:
        ncycle += 1
        pmat = calc_pmat(chem1, fmat, smat)
        norm_new = LA.norm(pmat)
        fmat = calc_fock(chem1, hcore, vee, pmat)
        if abs(norm_last - norm_new) < chem1.pmat_cutoff:
            break
        elif ncycle == 1000:
            print('Error -- Max SCF Cycle Reached -- Exiting...')
            sys.exit()
        else:
            norm_last = norm_new

    # AO to MO Transformation
    hcoremo, veemo = tran.transform_ao2mo(chem1, fmat, smat, hcore, vee)
    e_hf = tran.calc_ehf(chem1, hcoremo, veemo)

    hcorespin, veespin = tran.mo_space_spin_transform(chem1, hcoremo, veemo)
    orb_eng = tran.orb_eng_space_spin_transform(nspace, nspin, fmat, smat)

    chem1.set_hcore(hcorespin)
    chem1.set_vee(veespin)
    chem1.set_e_ref(e_hf)
    chem1.set_orb_eng(orb_eng)

    return chem1




def calc_pmat(sys_param, fmat, smat):
    evec, cvec = LA.eigh(fmat, smat)
    nspace = sys_param.get_nspace()
    nocc   = sys_param.get_nocc()
    pmat = NP.zeros((nspace, nspace), dtype=NP.float64)
    for mu in range(nspace):
        for nu in range(nspace):
            for a in range(nocc // 2): 
                pmat[mu,nu] += cvec[mu,a] * cvec[nu,a] * 2.0
    return pmat


def calc_fock(sys_param, hcore, vee, pmat):
    nspace = sys_param.get_nspace()
    nocc   = sys_param.get_nocc()
    fmat = NP.zeros((nspace, nspace), dtype=NP.float64)
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















    
    
