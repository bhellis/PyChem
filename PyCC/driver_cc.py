import numpy as NP
from ChemSys import chem_sys
from PyMP2 import driver_mp2 as MP2
from scipy import linalg as LA


def driver_cc(my_sys):
    nocc  = my_sys.get_nocc()
    nspin = my_sys.get_nspin()
    hcore = my_sys.get_hcore()
    vee   = my_sys.get_vee()
    orb_eng = my_sys.get_orb_eng()

    t2 = MP2.calc_t_mp2(nocc, nspin, hcore, vee, orb_eng)
    t1 = NP.zeros((nocc, nspin), dtype=NP.float64)
    #t2 = optimize_t2(nocc, nspin, vee, orb_eng, t2, chem1.pmat_cutoff)
    my_sys.set_e_corr(calc_cc_ecorr(nocc, nspin, vee, t1, t2))


def calc_cc_ecorr(nocc, nspin, vee, t1, t2):
    #leaving out FOCK for now
    ecorr = 0.0
    for i in range(nocc):
        for j in range(nocc):
            for a in range(nocc, nspin):
                for b in range(nocc, nspin):
                    t_value = t2[i,j,a,b] + (t1[i,a] * t1[j,b]) - (t1[i,b] * t1[j,a])
                    ecorr += (vee[i,a,j,b] - vee[i,b,j,a]) * t_value
    return 0.25 * ecorr


def optimize_t2(nocc, nspin, vee, orb_eng, t2, cutoff):
    norm_t2a = LA.norm(t2)
    norm_t2b = norm_t2a * 100.0
    while abs(norm_t2a - norm_t2b) > cutoff:
        norm_t2a = LA.norm(t2)
        t2new = calc_t2new(nocc, nspin, vee, orb_eng, t2)
        norm_t2b = LA.norm(t2ew)
        t2 = t2new
    return t2


def calc_t2new(nocc, nspin, vee, orb_eng, t2):
    t2new = np.zeros((nocc, nocc, nspin, nspin), dtype = NP.float64)
    for i in range(nocc):
        for j in range(nocc):
            for a in range(nocc, nspin):
                for b in range(nocc, nspin):
                    orb_eng_diff = orb_eng[i] + orb_eng[j] - orb_eng[a] - orb_eng[b]
                    t2new[i,j,a,b] = calc_numerator(i, j, a, b, nocc, nspin, vee, t2) / orb_eng_diff
    return t2new


def calc_numerator(i, j, a, b, nocc, nspin, vee, t2):
    ## NEEDS TO BE CHEMIST NOTATION!!!!!
    this_val = 0.0
    this_val += asymm_integral(i,j,a,b)
    for c in range(nocc, nspin):
        for d in range(nocc, nspin):
            this_val += asymm_integral(a,b,c,d) * t2[i,j,c,d] * 0.5
    for k in range(nocc):
        for l in range(nocc):
            this_val += asymm_integral(i,j,k,l) * t2[k,l,a,b] * 0.5
    for k in range(nocc):
        for c in range(nocc, nspin):
            #this_sum -= asymm_integral(b
            pass


def asymm_integral(i, j, a, b, vee):
    return vee[i,a,j,b] - vee[i,b,j,a]






                    


