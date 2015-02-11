import sys
import copy

import numpy as NP
from ChemSys import chem_sys
from PyMP2 import driver_mp2 as MP2
from scipy import linalg as LA
from Misc import timing

from PyCC import cc_routines

@timing.time_fxn
def driver_cc(chem1):
    nocc  = chem1.get_nocc()
    nspin = chem1.get_nspin()
    hcore = chem1.get_hcore()
    vee   = chem1.get_vee()
    orb_eng = chem1.get_orb_eng()

    t2 = MP2.calc_t_mp2(nocc, nspin, hcore, vee, orb_eng)
    t1 = NP.zeros((nocc, nspin), dtype=NP.float64)
    t2final = optimize_t2(nocc, nspin, vee, orb_eng, t1, t2, chem1.pmat_cutoff)
    chem1.set_e_corr(calc_cc_ecorr_fast(nocc, nspin, vee, t1, t2final))

    chem1.show_e_info('CCD')


def calc_cc_ecorr_fast(nocc, nspin, vee, t1, t2):
    ecorr = 0.0
    for i in range(nocc):
        for j in range(i + 1, nocc):
            for a in range(nocc, nspin):
                for b in range(a + 1, nspin):
                    t_value = t2[i,j,a,b] + (t1[i,a] * t1[j,b]) - (t1[i,b] * t1[j,a])
                    ecorr += asymm_integral(i, a, j, b, vee) * t_value
    return ecorr

def calc_cc_ecorr(nocc, nspin, vee, t1, t2):
    ecorr = 0.0
    for i in range(nocc):
        for j in range(nocc):
            for a in range(nocc, nspin):
                for b in range(nocc, nspin):
                    t_value = t2[i,j,a,b] + (t1[i,a] * t1[j,b]) - (t1[i,b] * t1[j,a])
                    ecorr += asymm_integral(i, a, j, b, vee) * t_value
    return 0.25 * ecorr


def optimize_t2(nocc, nspin, vee, orb_eng, t1, t2, cutoff):
    t2new = NP.zeros((nocc, nocc, nspin, nspin), dtype = NP.float64)
    precomp_vee = precomp_vee_vals(nocc, nspin, vee)
    ncycle = 0
    while True:
        ncycle += 1
        #print(ncycle)
        if ncycle == 100:
            print(' -- CCD MAX LOOP REACHED -- EXITING -- ')    
            sys.exit()
        cc_routines.cy_calc_t2new(nocc, nspin, precomp_vee, orb_eng, t2, t2new)
        #t2new = cc_routines.cy_calc_t2new(nocc, nspin, precomp_vee, orb_eng, t2, t2new)
        #t2new = calc_t2new(nocc, nspin, precomp_vee, orb_eng, t2, t2new)
        if abs(LA.norm(t2new) - LA.norm(t2)) < cutoff:
            break
        t2 = copy.deepcopy(t2new)
    return t2new


def calc_t2new(nocc, nspin, precomp_vee, orb_eng, t2, t2new):
    for i in range(nocc):
        #for j in range(nocc):
        for j in range(i + 1, nocc):
            for a in range(nocc, nspin):
                #for b in range(nocc, nspin):
                for b in range(a + 1, nspin):
                    if abs(precomp_vee[i,a,j,b]) < 1.0e-8:
                        continue
                    orb_eng_diff = orb_eng[i] + orb_eng[j] - orb_eng[a] - orb_eng[b]
                    t2new[i,j,a,b] = cc_routines.cy_calc_numerator(i, j, a, b, nocc, nspin, precomp_vee, t2) / orb_eng_diff
                    #t2new[i,j,a,b] = calc_numerator(i, j, a, b, nocc, nspin, precomp_vee, t2) / orb_eng_diff
                    t2new[j,i,a,b] = -t2new[i,j,a,b]
                    t2new[j,i,b,a] = t2new[i,j,a,b]
                    t2new[i,j,b,a] = -t2new[i,j,a,b]
    return t2new


def calc_numerator(i, j, a, b, nocc, nspin, precomp_vee, t2):
    this_val = 0.0
    this_val += precomp_vee[i,a,j,b]
    #this_val += asymm_integral(a,i,b,j,vee)
    for c in range(nocc, nspin):
        for d in range(nocc, nspin):
            this_val += precomp_vee[a,c,b,d] * t2[i,j,c,d] * 0.5
            #this_val += asymm_integral(a,c,b,d,vee) * t2[i,j,c,d] * 0.5
    for k in range(nocc):
        for l in range(nocc):
            this_val += precomp_vee[i,k,j,l] * t2[k,l,a,b] * 0.5
            #this_val += asymm_integral(i,k,j,l,vee) * t2[k,l,a,b] * 0.5
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
                    #this_val += asymm_integral(k,c,l,d,vee) * \
                    this_val += precomp_vee[k,c,l,d] * \
                                ((0.25 * t2[i,j,c,d] * t2[k,l,a,b]) \
                                - (0.5  * (t2[i,j,a,c] * t2[k,l,b,d]) + (t2[i,j,b,d] * t2[k,l,a,c])) \
                                - (0.5  * (t2[i,k,a,b] * t2[j,l,c,d]) + (t2[i,k,c,d] * t2[j,l,a,b])) \
                                + ((t2[i,k,a,c] * t2[j,l,b,d]) + (t2[i,k,b,d] * t2[j,l,a,c])))
    return this_val
     


def asymm_integral(i, a, j, b, vee):
    #              ^1 ^1 ^2 ^2
    return vee[i,a,j,b] - vee[i,b,j,a]


def precomp_vee_vals(nocc, nspin, vee):
    precomputed = NP.zeros((nspin, nspin, nspin, nspin), dtype = NP.float64) 
    for i in range(nspin):
        for j in range(nspin):
            for a in range(nspin):
                for b in range(nspin):
                    precomputed[i,a,j,b] = asymm_integral(i, a, j, b, vee)
    return precomputed






                    


