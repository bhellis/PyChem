import numpy as NP
from ChemSys import chem_sys
from Misc import timing


@timing.time_fxn
def driver_mp2(chem1):
    nocc = chem1.get_nocc()
    nspin = chem1.get_nspin()
    hcore = chem1.get_hcore()
    vee = chem1.get_vee()
    orb_eng = chem1.get_orb_eng()

    t_mp2 = calc_t_mp2(nocc, nspin, hcore, vee, orb_eng)
    e_corr = calc_mp2_corr(nocc, nspin, vee, t_mp2)
    #print(e_corr)
    chem1.set_e_corr(e_corr)

    chem1.show_e_info('MP2')


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


