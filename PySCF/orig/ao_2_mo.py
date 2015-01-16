import numpy as np
from scipy import linalg
from PySCF import chem_sys


def transform_ao2mo(sysparam, fmat, smat, hcore, vee):
    evec, cvec = linalg.eigh(fmat, smat)
    return ao2mo_hcore(sysparam, cvec, hcore), ao2mo_vee(sysparam, cvec, vee)
    #return ao2mo_hcore(sysparam, cvec, hcore), ao2mo_vee_slow(sysparam, cvec, vee)


def ao2mo_hcore(sysparam, cvec, hcore):
    nspace = sysparam.get_nspace()
    hcore_mo = np.zeros((nspace, nspace), dtype = np.float64)
    for p in range(nspace):               ## mo idx
        for q in range(nspace):           ## mo idx
            for mu in range(nspace):      ## ao idx
                for nu in range(nspace):  ## ao idx
                    hcore_mo[p,q] += (cvec[mu,p] * hcore[mu,nu] * cvec[nu,q])
                    hcore_mo[q,p] = hcore_mo[p,q]
    return hcore_mo


def ao2mo_vee_slow(sysparam, cvec, vee):
    nspace = sysparam.get_nspace()
    veemo = np.zeros((nspace, nspace, nspace, nspace), dtype = np.float64)
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
    veemo = np.zeros((nspace, nspace, nspace, nspace), dtype = np.float64)
    temp1 = np.zeros((nspace, nspace, nspace, nspace), dtype = np.float64)
    temp2 = np.zeros((nspace, nspace, nspace, nspace), dtype = np.float64)
    temp3 = np.zeros((nspace, nspace, nspace, nspace), dtype = np.float64)
    #temp4 = np.zeros((nspace, nspace, nspace, nspace), dtype = np.float64)
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
    ## ERRORS
    nocc = sysparam.get_nocc()
    xsum1body = xsum2body = 0.0 
    for i in range(nocc // 2):
        xsum1body += 2.0 * hcore[i,i]
        for j in range(nocc // 2):
            xsum2body += (2.0 * vee[i,i,j,j]) - vee[i,j,j,i]
    #print('1 body: %s, 2 body: %s' % (xsum1body, xsum2body))
    return xsum1body + xsum2body











