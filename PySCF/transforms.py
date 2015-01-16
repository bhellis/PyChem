import numpy as NP
from scipy import linalg
from ChemSys import chem_sys


def transform_ao2mo(sysparam, fmat, smat, hcore, vee):
    evec, cvec = linalg.eigh(fmat, smat)
    return ao2mo_hcore(sysparam, cvec, hcore), ao2mo_vee(sysparam, cvec, vee)


def ao2mo_hcore(sysparam, cvec, hcore):
    nspace = sysparam.get_nspace()
    hcore_mo = NP.zeros((nspace, nspace), dtype = NP.float64)
    for p in range(nspace):               ## mo idx
        for q in range(nspace):           ## mo idx
            for mu in range(nspace):      ## ao idx
                for nu in range(nspace):  ## ao idx
                    hcore_mo[p,q] += (cvec[mu,p] * hcore[mu,nu] * cvec[nu,q])
                    hcore_mo[q,p] = hcore_mo[p,q]
    return hcore_mo


def ao2mo_vee_slow(sysparam, cvec, vee):
    nspace = sysparam.get_nspace()
    veemo = NP.zeros((nspace, nspace, nspace, nspace), dtype = NP.float64)
    for p in range(nspace):                                     ## mo idx
        for q in range(nspace):                                 ## mo idx
            for r in range(nspace):                             ## mo idx
                for s in range(nspace):                         ## mo idx
                    for mu in range(nspace):                    ## ao idx                    
                        for nu in range(nspace):                ## ao idx
                            for rho in range(nspace):           ## ao idx
                                for sgma in range(nspace):      ## ao idx
                                    veemo[p,q,r,s] += cvec[mu,p] * cvec[nu,q] * vee[mu,nu,rho,sgma] * cvec[rho,r] * cvec[sgma,s]
    return veemo

def ao2mo_vee(sysparam, cvec, vee):
    nspace = sysparam.get_nspace()
    veemo = NP.zeros((nspace, nspace, nspace, nspace), dtype = NP.float64)
    temp1 = NP.zeros((nspace, nspace, nspace, nspace), dtype = NP.float64)
    temp2 = NP.zeros((nspace, nspace, nspace, nspace), dtype = NP.float64)
    temp3 = NP.zeros((nspace, nspace, nspace, nspace), dtype = NP.float64)
    for s in range(nspace):
        for mu in range(nspace):
            for nu in range(nspace):
                for rho in range(nspace):
                    for sgma in range(nspace):
                        temp1[mu,nu,rho,s] += vee[mu,nu,rho,sgma] * cvec[sgma,s]
    for r in range(nspace):
        for s in range(nspace):
            for mu in range(nspace):
                for nu in range(nspace):
                    for rho in range(nspace):
                        temp2[mu,nu,r,s] += temp1[mu,nu,rho,s] * cvec[rho,r]
    for q in range(nspace):
        for r in range(nspace):
            for s in range(nspace):
                for mu in range(nspace):
                    for nu in range(nspace):
                        temp3[mu,q,r,s] += temp2[mu,nu,r,s] * cvec[nu,q]
    for p in range(nspace):
        for q in range(nspace):
            for r in range(nspace):
                for s in range(nspace):
                    for mu in range(nspace):
                        veemo[p,q,r,s] += temp3[mu,q,r,s] * cvec[mu,p]
    return veemo


def calc_ehf(sysparam, hcore, vee):
    nocc = sysparam.get_nocc()
    xsum1body = xsum2body = 0.0 
    for i in range(nocc // 2):
        xsum1body += 2.0 * hcore[i,i]
        for j in range(nocc // 2):
            xsum2body += (2.0 * vee[i,i,j,j]) - vee[i,j,j,i]
    return xsum1body + xsum2body


def mo_space_spin_transform(sysparam, hcore, vee):
    nspace = sysparam.get_nspace()
    nspin = sysparam.get_nspin()
    spinmat = make_spinmat(nspace)
    hcore_spin = NP.zeros((nspin, nspin), dtype = NP.float64)
    vee_spin = NP.zeros((nspin, nspin, nspin, nspin), dtype = NP.float64)
    for ispc in range(nspace):
        for ispin in range(2):
            imo = spinmat[ispc,ispin]
            for jspc in range(nspace):
                for jspin in range(2):
                    jmo = spinmat[jspc,jspin]
                    if ispin != jspin:
                        continue
                    hcore_spin[imo,jmo] = hcore[ispc,jspc]
                    for kspc in range(nspace):
                        for kspin in range(2):
                            kmo = spinmat[kspc,kspin]
                            for lspc in range(nspace):
                                for lspin in range(2):
                                    lmo = spinmat[lspc,lspin]
                                    if kspin != lspin:
                                        continue
                                    vee_spin[imo,jmo,kmo,lmo] = vee[ispc,jspc,kspc,lspc]
    return hcore_spin, vee_spin
            


def make_spinmat(nspace):
    spinmat = NP.zeros((nspace, 2), dtype = int)
    imo = -1
    for ispc in range(nspace):
        for ispn in range(2):
            imo += 1
            spinmat[ispc,ispn] = imo
    return spinmat


def spin_eng(mysys, hcorespin, veespin):
    nocc = mysys.get_nocc()
    eng1 = 0.0
    eng2 = 0.0
    for i in range(nocc):
        eng1 += hcorespin[i,i]
        for j in range(nocc):
            eng2 += 0.5 * (veespin[i,i,j,j] - veespin[i,j,j,i]) 
    return eng1 + eng2
            

def orb_eng_space_spin_transform(nspace, nspin, fmat, smat):
    evec, cvec = linalg.eigh(fmat, smat)
    orb_engspin = NP.zeros((nspin), dtype = NP.float64)
    for i in range(nspace):
        orb_engspin[2 * i] = evec[i]
        orb_engspin[(2 * i) + 1] = evec[i]
    return orb_engspin
        
            










