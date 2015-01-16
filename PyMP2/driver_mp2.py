import numpy as NP
from ChemSys import chem_sys


def driver_mp2(my_sys):
    nocc = my_sys.get_nocc()
    nspin = my_sys.get_nspin()
    hcore = my_sys.get_hcore()
    vee = my_sys.get_vee()
    orb_eng = my_sys.get_orb_eng()

    t_mp2 = calc_t_mp2(nocc, nspin, hcore, vee, orb_eng)
    e_corr = calc_mp2_corr(nocc, nspin, vee, t_mp2)
    #print(e_corr)
    my_sys.set_e_corr(e_corr)


def calc_mp2_corr(nocc, nspin, vee, t_mp2):
    ecorr = 0.0
    for i in range(nocc):
        for j in range(nocc):
            for a in range(nocc, nspin):
                for b in range(nocc, nspin):
                    ecorr += (vee[i,a,j,b] - vee[i,b,j,a]) * t_mp2[i,j,a,b]
    return ecorr * 0.25


def calc_t_mp2(nocc, nspin, hcore, vee, orb_eng):
    t_mp2 = NP.zeros((nocc, nocc, nspin, nspin), dtype = NP.float64)
    for i in range(nocc):
        for j in range(nocc):
            for a in range(nocc, nspin):
                for b in range(nocc, nspin):
                    t_mp2[i,j,a,b] = (vee[i,a,j,b] - vee[i,b,j,a]) / \
                                     (orb_eng[i] + orb_eng[j] - orb_eng[a] - orb_eng[b])
    return t_mp2


